from util.time_generation import TimeGeneration
import os

SERVER_ADDR= 'localhost'   # When running in a real distributed setting, change to the server's IP address
SERVER_PORT = 51000

person_detection_datasets = ["fomaml_datasets/Garden", "fomaml_datasets/Dining"]

#FCLayer parameters
input_sz = 256
output_sz = 2
inner_learning_rate = 0.01
outer_learning_rate = 0.001
inner_epoch = 3

#maximum number of global aggregation rounds
MAX_GLOBAL_ROUND = 300

#number of local rounds before global aggregation
tau_setup = 1 

#number of virtual clients
n_nodes = 2 

#number of physical clients
serial_nodes = 2 

tau_max = 100  # Specifies the maximum value of tau
