# FOMAML Federated Learning for person detection in different environments — Client & Project Overview

This README documents the client and server scripts, dataset layout, and helper modules in this workspace. It is intended to help developers understand which scripts to run for virtual clients and the server, and where personal datasets and FC-layer/data extraction functions live.

## Setup

1. Arduino IDE 2.0.4 on Linux

2. Google Coral Micro Dev Board setup on Arduino IDE
   - The installed Coral library does not include serial input, which is required for supervised learning.
   - A Coral package with serial input support is provided in `coral_arduino_package/` (the Coral package version should be 1.0.1).
   - Replace the existing bundled `.a` Coral package file in the Arduino library after installing the Coral library package. It is recommended to back up the pre-existing file first.
   - Example (adjust to your runtime and actual filename/extension):

     ```bash
     cp coral_arduino_package/liblibs_arduino_coral_micro_bundled.a \
     ~/.arduino15/packages/coral/hardware/coral_micro/1.0.1/variants/coral_micro/libs/
     ```

3. Activate virtual environment when using python scripts (activate in root folder)
  
## Overview

- Client scripts:
  - `client_fomaml.py` — primary client script used for FOMAML experiments.
  - `coral_fomaml.py` — client script for the Google Coral Micro Dev Board
  - `nano_fomaml.py` — client script for the Arduino Nano BLE
- Server script:
  - `server_fomaml.py` — coordinates federated rounds and aggregates client updates.
- Configuration:
  - `config_fomaml.py` — central configuration script/file for experiments (e.g., hyperparameters, paths).
- Datasets:
  - `fomaml_datasets/` — contains personal datasets for virtual clients. Two example datasets provided:
    - `dining`
    - `garden`
- Helper functions:
  - `fomaml/` — contains functions for the fully-connected (FC) layer and data extraction utilities.
  - `util/` — contains functions used by the server and clients to communicate with each other.
  - `get_image/` — contains two scripts to view the images taken by the Google Coral Micro Dev Board and Arduino Nano BLE

- Model:
  - `tflite_model/` — contains the tflite model used by the Coral and Nano.

- Results:
  - `results/` — results on the accuracies and losses of each clients are plotted across all global rounds and stored in this folder.

- Arduino Scripts:
  - `results/ArduinoBLE_Federated_FOMAML` - contains the Arduino script, FC layer script and tflite model to implement federated FOMAML on the Arduino Nano BLE.
  - `results/coral_Federated_FOMAML` - contains the Arduino script, FC layer script and tflite model to implement federated FOMAML on the Google Coral Micro Dev Board.

## Quick start

1. Prepare configuration

   - Edit the `config_fomaml` script/file to set experiment parameters (learning rate, maximum global rounds, dataset paths, etc.).

2. Start the server

   - Run the `server_fomaml` script to start the federated server that coordinates clients.
     Example (adjust to your runtime and actual filename/extension):

     ```bash
     # Example — replace with the real filename/command
     python server_fomaml.py
     ```

3. Start the clients

   - Launch virtual clients first using separate terminals.
     Example (adjust to your runtime and actual filename/extension):
     ```bash
     python client_fomaml.py
     ```

   - Once the server prints "All virtual clients connected", run the coral and nano scripts in separate terminals:
     Example (adjust to your runtime and actual filename/extension):
     ```bash
     python coral_fomaml.py
     python nano_fomaml.py
     ```

   - Notes for physical devices when plugged in PC/Laptop:
     - Arduino Nano BLE: after connecting the device to the host, wait for the user LED to turn yellow before running the Nano script.
     - Google Coral Micro: wait for the camera LED to flash about three times and then about 5 seconds before running the Coral script (serial setup can take 20–30 seconds to stabilize).

   - After connecting to the Google Coral Micro and Arduino Nano BLE, type the character 'y' to both devices via their terminals to signal them to connect to the server. Each device should print lines similar to:
     ```
     [Nano] FOMAML Simulation round 0
     [Coral] FOMAML Simulation round 0
     ```

   - The server will print "All physical devices connected" and then start any preprocessing scripts to fetch datasets for the virtual clients.

   - When the server prints "Start learning", clients may begin performing local rounds.

4. Datasets

   - Personal/virtual-client datasets are stored under `fomaml_datasets/`.
   - Two example datasets in this workspace:
     - `dining` — personal data for the "dining" virtual client
     - `garden` — personal data for the "garden" virtual client

5. Helper utilities

   - The `fomaml/` folder contains functions related to the FC layer and data extraction that client scripts import and use. Inspect that folder to see available helpers for model building and dataset preprocessing.

## Training notes

1. Supervised learning using the Google Coral Micro Dev Board and Arduino Nano BLE
   - For FOMAML, there is a support set and a query set, so the total number of images taken for a local round is the sum of the sizes of both sets.
   - Use serial input to label the images taken: use `y` for person and `n` for non-person, and the photo is taken subsequently after that.
   - The user LED on both boards can be used as an indicator of whether the capture-image process is fully completed.
     - For Arduino Nano BLE: Green = Person, Red = Non-Person
     - For Google Coral Micro: Green = Person, No light = Non-Person

2. Serial issues with the Google Coral Micro Dev Board and Arduino Nano BLE
   - Wait for the expected messages sent from the boards to fully print on the terminals before sending inputs to the boards; otherwise, it may cause the serial port to crash.
   - When the global model is sending weights to the board or vice versa, do not send inputs to the boards, or it may cause the serial port to hang and crash. Wait for the weight-receive or weight-sent message to be printed by both boards before sending input.

3. Scripts to aid in experimental setup
   - `get_image/imageCapture_Coral.py` captures the photo taken by the Coral for every image taken and saves it to the local folder.
   - Note (for coral): In `arduino_scripts/coral_Federated_FOMAML/image_provider.cpp`, refer to line 35 and use the commented function for capturing the image on the Google Coral Micro Dev Board when using the image test script. Make sure to comment out the `receiveGlobalWeights()` and `sendLocalWeights()` functions in `arduino_scripts/coral_Federated_FOMAML/coral_Federated_FOMAML.ino`.
   - Note (for nano): In `arduino_scripts/ArduinoBLE_Federated_FOMAML/ArduinoBLE_Federated_FOMAML.ino`, refer to line 96 and remove the commented sections in the `CaptureImage()` function for capturing the image on the Arduino Nano BLE when using the image test script. Make sure to comment out the `receiveGlobalWeights()` and `sendLocalWeights()` functions in `arduino_scripts/ArduinoBLE_Federated_FOMAML/ArduinoBLE_Federated_FOMAML.ino`.