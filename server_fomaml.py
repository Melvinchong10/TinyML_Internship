import socket
import time

import numpy as np
import random

from util.utils import send_msg, recv_msg

# Configurations are in a separate config.py file
from config_fomaml import *
from fomaml.collect_data import compute_supp_acc_graph, compute_supp_loss_graph, compute_query_before_acc_graph, compute_query_after_acc_graph, compute_query_before_loss_graph, compute_query_after_loss_graph

global_round = 0

clients_supp_acc = {}
clients_query_before_acc = {}
clients_query_after_acc = {}
clients_supp_loss = {}
clients_query_before_loss = {}
clients_query_after_loss = {}
clients_avr_local_time = {}

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all=[]
physical_client_sock_all = []

# Establish connections to each client, up to n_nodes clients
while len(client_sock_all) + len(physical_client_sock_all) < n_nodes + serial_nodes:
    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    msg = recv_msg(client_sock, 'MSG_CLIENT_TYPE')
    if(msg[1] == 'VIRTUAL'):
        client_sock_all.append(client_sock)
    elif(msg[1] == 'PHYSICAL'):
        physical_client_sock_all.append(client_sock)
    else:
        print("Unknown type of client")


w_global_init = np.random.randn(input_sz, output_sz) * 0.01
w_global = w_global_init

print(f"Weight[0][0] : {w_global[0][0]}, Weight[55][0] : {w_global[55][0]}, Weight[128][0] : {w_global[128][1]} ")
random.shuffle(person_detection_datasets)

for n in range(0, n_nodes):
    msg = ['MSG_INIT_SERVER_TO_CLIENT', person_detection_datasets[n]]
    print(msg)
    send_msg(client_sock_all[n], msg)

print('All virtual clients connected')

for s in range(0, serial_nodes):
    recv_msg(physical_client_sock_all[s], 'MSG_INIT_PHYSICAL_CLIENT_TO_SERVER')

print('All physical devices connected')

# Wait until all clients complete data preparation and sends a message back to the server
for n in range(0, n_nodes):
    recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')

print('Start learning')

time_global_aggregation_all = None

total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
                            # using predefined values when use_fixed_averaging_slots = true
#it_each_local = None
#it_each_global = None

is_last_round = False
#is_eval_only = False

#tau_new_resume = None

# Loop for multiple rounds of local iterations + global aggregation
while True:
    print(f"Global round: {global_round}")
    print('---------------------------------------------------------------------------')

    print('current tau config:', tau_setup)

    if(global_round == MAX_GLOBAL_ROUND):
        is_last_round = True

    time_total_all_start = time.time()

    for n in range(0, n_nodes):
        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_setup, is_last_round]
        send_msg(client_sock_all[n], msg)

    for s in range(0, serial_nodes):
        msg = ['MSG_WEIGHT_TAU_SERVER_TO_PHYSICAL_CLIENT', w_global, tau_setup, is_last_round]
        send_msg(physical_client_sock_all[s], msg)

    w_global_prev = w_global

    print('Waiting for local iteration at client')

    w_global = np.zeros((input_sz, output_sz))
    loss_last_global = 0.0
    loss_w_prev_min_loss = 0.0
    received_loss_local_w_prev_min_loss = False
    data_size_total = 0
    time_all_local_all = 0
    data_size_local_all = []

   # tau_actual = 0

    for n in range(0, n_nodes):
        msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
        w_local = msg[1]
        time_all_local = msg[2]
        data_size_local = msg[3]

        w_global += w_local * data_size_local
        data_size_local_all.append(data_size_local)
        data_size_total += data_size_local
        time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node

    for s in range(0, serial_nodes):
        msg = recv_msg(physical_client_sock_all[s], 'MSG_WEIGHT_TIME_SIZE_PHYSICAL_CLIENT_TO_SERVER')
        w_local = msg[1]
        time_all_local = msg[2]
        data_size_local = msg[3]

        w_global += w_local * data_size_local
        data_size_local_all.append(data_size_local)
        data_size_total += data_size_local


    w_global /= data_size_total

    if True in np.isnan(w_global):
        print('*** w_global is NaN, using previous value')
        w_global = w_global_prev   # If current w_global contains NaN value, use previous w_global
        use_w_global_prev_due_to_nan = True
    else:
        use_w_global_prev_due_to_nan = False

    # Calculate time
    time_total_all_end = time.time()
    time_total_all = time_total_all_end - time_total_all_start
    time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

    print('Time for one local iteration:', time_all_local_all / tau_setup)
    print('Time for global averaging:', time_global_aggregation_all)

    global_round += 1

    if(is_last_round == True):
        print("Collecting client's final data")
        for n in range(0, n_nodes):
            msg = recv_msg(client_sock_all[n], 'MSG_CLIENT_DATA_TO_SERVER')
            client_name = msg[1]
            client_supp_acc = msg[2]
            client_query_before_acc = msg[3]
            client_query_after_acc = msg[4]
            client_supp_loss = msg[5]
            client_query_before_loss = msg[6]
            client_query_after_loss = msg[7]
            client_avr_local_time = msg[8]

            clients_supp_acc[client_name] = client_supp_acc
            clients_query_before_acc[client_name] = client_query_before_acc
            clients_query_after_acc[client_name] = client_query_after_acc
            clients_supp_loss[client_name] = client_supp_loss
            clients_query_before_loss[client_name] = client_query_before_loss
            clients_query_after_loss[client_name] = client_query_after_loss
            clients_avr_local_time[client_name] = client_avr_local_time

        for s in range(0, serial_nodes):
            msg = recv_msg(physical_client_sock_all[s], 'MSG_CLIENT_DATA_TO_SERVER')
            client_name = msg[1]
            client_supp_acc = msg[2]
            client_query_before_acc = msg[3]
            client_query_after_acc = msg[4]
            client_supp_loss = msg[5]
            client_query_before_loss = msg[6]
            client_query_after_loss = msg[7]
            client_avr_local_time = msg[8]

            clients_supp_acc[client_name] = client_supp_acc
            clients_query_before_acc[client_name] = client_query_before_acc
            clients_query_after_acc[client_name] = client_query_after_acc
            clients_supp_loss[client_name] = client_supp_loss
            clients_query_before_loss[client_name] = client_query_before_loss
            clients_query_after_loss[client_name] = client_query_after_loss
            clients_avr_local_time[client_name] = client_avr_local_time

        print("Received all virtual and physical clients data")
        print("Plotting graphs using the collected data")
        compute_supp_acc_graph(clients_supp_acc)
        compute_query_before_acc_graph(clients_query_before_acc)
        compute_query_after_acc_graph(clients_query_after_acc)
        compute_supp_loss_graph(clients_supp_loss)
        compute_query_before_loss_graph(clients_query_before_loss)
        compute_query_after_loss_graph(clients_query_after_loss)

        for n in range(0, n_nodes):
            msg = ['FINAL_DATA_RECEIVED']
            send_msg(client_sock_all[n], msg)

        for s in range(0, serial_nodes):
            msg = ['FINAL_DATA_RECEIVED']
            send_msg(physical_client_sock_all[s], msg)

        print("Clients local round average training time: ")
        for client_name, local_time in clients_avr_local_time.items():
            print(f"{client_name}: {local_time: .10f}s") 
        print("End of federated learning simulation")

        break
            