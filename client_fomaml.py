import socket
import time
import struct

from util.utils import send_msg, recv_msg

# Configurations are in a separate config.py file
from config_fomaml import *
from fomaml.FCLayer import FCLayerSim
from fomaml.preprocess_dataset import get_persondetect_data
from fomaml.collect_data import compute_loss_graph, compute_acc_graph

sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

msg = ['MSG_CLIENT_TYPE','VIRTUAL']
send_msg(sock, msg)


batch_size_prev = None
total_data_prev = None
sim_prev = None

local_supp_loss = []
local_supp_acc = []
local_query_before_loss = []
local_query_before_acc = []
local_query_after_loss = []
local_query_after_acc = []
total_local_time = 0.0
avr_local_time = 0.0
global_round = 0
supp_sz = 2
query_sz = 2

try:
    while True:
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')

        dataset = msg[1]

        if(dataset == "fomaml_datasets/Garden"):
            client_name = "garden"

        if(dataset == "fomaml_datasets/Dining"):
            client_name = "dining"

        model = FCLayerSim(input_size = 256, output_size= 2, 
                            inner_lr =inner_learning_rate, outer_lr = outer_learning_rate, supp_sz = supp_sz,
                            query_sz = query_sz, L2_lambda= 0.0001)
        
        #get data from provided dataset path
        (X_person_train, X_non_person_train, X_person_test, X_non_person_test,
        y_person_train, y_nonperson_train, y_person_test, y_nonperson_test) = get_persondetect_data(dataset)
        
        print("About to send DATA_PREP_FINISHED")
        msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
        send_msg(sock, msg)
        print("DATA_PREP_FINISHED sent")

        while True:
            print('---------------------------------------------------------------------------')

            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
            w_global = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]

            time_local_start = time.time()  #Only count this part as time for local iteration because the remaining part does not increase with tau

            for i in range(0, tau_config):
                if i == 0:
                    w_local, supp_acc, supp_loss, query_before_acc, query_before_loss, query_after_acc, query_after_loss = model.FOMAML_simulation_round(X_person_train_feat=X_person_train, 
                                                        X_nonperson_train_feat=X_non_person_train, y_person_train_oh=y_person_train,
                                                        y_nonperson_train_oh=y_nonperson_train, local_epoch= inner_epoch,global_w=w_global)
                    local_supp_acc.append(supp_acc)
                    local_supp_loss.append(supp_loss)
                    local_query_before_acc.append(query_before_acc)
                    local_query_before_loss.append(query_before_loss) 
                    local_query_after_acc.append(query_after_acc)
                    local_query_after_loss.append(query_after_loss) 
                else:
                    w_local, supp_acc, supp_loss, query_before_acc, query_before_loss, query_after_acc, query_after_loss = model.FOMAML_simulation_round(X_person_train_feat=X_person_train, 
                                                        X_nonperson_train_feat=X_non_person_train, y_person_train_oh=y_person_train,
                                                        y_nonperson_train_oh=y_nonperson_train, local_epoch= inner_epoch, global_w= None)
                    
            # Local operation finished, global aggregation starts
            time_local_end = time.time()
            time_all_local = time_local_end - time_local_start
            total_local_time += time_all_local
            global_round += 1
            print('time_all_local =', time_all_local)
            
            data_size_local = supp_sz + query_sz
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w_local, time_all_local, data_size_local]
            send_msg(sock, msg)

            if is_last_round:
                print("[Client] Sending final data to server")
                compute_loss_graph(supp_loss= local_supp_loss, query_before_loss= local_query_before_loss, query_after_loss= local_query_after_loss, client_type= client_name)
                compute_acc_graph(supp_acc= local_supp_acc, query_before_acc= local_query_before_acc, query_after_acc= local_query_after_acc, client_type= client_name)
                avr_local_time = total_local_time / global_round
                msg = ['MSG_CLIENT_DATA_TO_SERVER', client_name, local_supp_acc, local_query_before_acc, local_query_after_acc, local_supp_loss, local_query_before_loss, local_query_after_loss, avr_local_time]
                send_msg(sock, msg)
                msg = recv_msg(sock, 'FINAL_DATA_RECEIVED')
                print("End of federated learning simulation")
                break

except (struct.error, socket.error):
    print('Server has stopped')
    pass
