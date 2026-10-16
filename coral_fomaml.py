import socket
import numpy as np
import serial
import threading
import time
import queue
from config_fomaml import *
from util.utils import send_msg, recv_msg
from fomaml.collect_data import compute_loss_graph, compute_acc_graph

serial_queue = queue.Queue()

# Thread to send keyboard input to the Coral
def keyboard_thread():
    while True:
        try:
            text = input()
            ser.write((text + "\n").encode())
        except KeyboardInterrupt:
            break

def serial_reader():
    global receive_w
    """The ONLY function that reads from the Coral serial port."""
    print("[Serial Reader] STARTED")
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                serial_queue.put(line)
                if not receive_w:
                    print(line)

        except serial.SerialException as e:
            print("[Serial Reader] Serial error:", e)
            break

        except Exception as e:
            print("[Serial Reader] Error:", e)
            break
PORT = "/dev/ttyACM0"
BAUD = 115200
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

msg = ['MSG_CLIENT_TYPE','PHYSICAL']
send_msg(sock, msg)

is_last_round = False
sending_w = False
receive_w = False
local_supp_loss = []
local_supp_acc = []
local_query_before_loss = []
local_query_before_acc = []
local_query_after_loss = []
local_query_after_acc = []
x_weights_buff = []
total_local_time = 0.0
avr_local_time = 0.0
global_round = 0
supp_sz = 2
query_sz = 2
local_data_size = supp_sz + query_sz

try:
    ser = serial.Serial(PORT, BAUD)
    print("[Client] Successfully connected to Coral")

except serial.SerialException as e:
    print("[Client] Failed to connect:", e)

print("[Client] Begin supervised learning")
threading.Thread(target=keyboard_thread, daemon=True).start()
threading.Thread(
    target=serial_reader,
    daemon=True
).start()

while True:
    try:
        line = serial_queue.get(timeout=0.1)

        if line == "[Coral] Starting local FOMAML round...":
            time_local_start = time.time() 
            continue

        if line[0:30] == "[Coral] Average support loss: ":
            supp_loss = float(line[30:])
            local_supp_loss.append(supp_loss)
            continue

        if line[0:35] == "[Coral] Average query before loss: ":
            query_before_loss = float(line[35:])
            local_query_before_loss.append(query_before_loss)
            continue

        if line[0:34] == "[Coral] Average query after loss: ":
            query_after_loss = float(line[34:])
            local_query_after_loss.append(query_after_loss)
            continue

        if line[0:34] == "[Coral] Average support accuracy: ":
            supp_acc = float(line[34:])
            local_supp_acc.append(supp_acc)
            continue

        if line[0:39] == "[Coral] Average query before accuracy: ":
            query_before_acc = float(line[39:])
            local_query_before_acc.append(query_before_acc)
            continue

        if line[0:38] == "[Coral] Average query after accuracy: ":
            query_after_acc = float(line[38:])
            local_query_after_acc.append(query_after_acc)
            continue

        if line == "CONNECT":
            msg = ['MSG_INIT_PHYSICAL_CLIENT_TO_SERVER']
            send_msg(sock, msg)
            continue

        if line == "BEGIN_WEIGHTS_TO_CLIENT":
            print("[Client] Receiving weights from Coral")
            receive_w = True
            rx_weights_buff = []
            continue

        if line == "END_WEIGHTS_TO_CLIENT":
            receive_w = False
            print("[Client] Completed receiving weights from Coral")

            w_local = np.array(rx_weights_buff).reshape(256, 2)

            #print("[Client] Weight shape:", w_local.shape)
            time_local_end = time.time()
            time_all_local = time_local_end - time_local_start
            total_local_time += time_all_local
            global_round += 1
            msg = ['MSG_WEIGHT_TIME_SIZE_PHYSICAL_CLIENT_TO_SERVER', w_local , time_all_local ,local_data_size]
            send_msg(sock, msg)
            if is_last_round:
                break
            continue
        
        if line == "REQ_WEIGHTS":
            sending_w = True
            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_PHYSICAL_CLIENT')
            w_global = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]

            #convert the (256,2) to 512 for coral 
            w_global = w_global.reshape(512)
            continue
        
        if receive_w == True:
            try:
                rx_weights_buff.append(float(line))
            except ValueError:
                pass
            continue

        
        if sending_w == True:
            ser.timeout = 2
            print("[Client] Starting to sent weights to coral..")
            ser.write(b"BEGIN_WEIGHTS_TO_CORAL\n")
            ser.flush()
            time.sleep(1)
            for index,w in enumerate(w_global):
                #print(f"[Client] Sending weight {index} : {w} ")
                ser.write(f"{w:.10f}\n".encode())
                ser.flush()
                time.sleep(0.005)


            ser.write(b"END_WEIGHTS_TO_CORAL\n")
            print("[Client] Finished sending weights")
            sending_w = False
            continue
    
    except queue.Empty:
        continue
    except KeyboardInterrupt:
        break

if is_last_round:
    print("[Coral] Sending final data to server")
    compute_loss_graph(supp_loss= local_supp_loss, query_before_loss= local_query_before_loss, query_after_loss= local_query_after_loss, client_type= "coral")
    compute_acc_graph(supp_acc= local_supp_acc, query_before_acc= local_query_before_acc, query_after_acc= local_query_after_acc, client_type= "coral")
    avr_local_time = total_local_time / global_round
    msg = ['MSG_CLIENT_DATA_TO_SERVER', "coral", local_supp_acc, local_query_before_acc, local_query_after_acc, local_supp_loss, local_query_before_loss, local_query_after_loss, avr_local_time]
    send_msg(sock, msg)
    msg = recv_msg(sock, 'FINAL_DATA_RECEIVED')
    print("End of federated learning simulation")

ser.close()