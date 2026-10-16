import serial
import numpy as np
from PIL import Image
import threading

PORT = "/dev/ttyACM0"
BAUD = 115200

WIDTH = 96
HEIGHT = 96

ser = serial.Serial(PORT, BAUD, timeout=0.1)

capturing = False
pixels = []
image_count = 0


# Thread to send keyboard input to the Coral
def keyboard_thread():
    while True:
        try:
            text = input()
            ser.write((text + "\n").encode())
        except KeyboardInterrupt:
            break


threading.Thread(target=keyboard_thread, daemon=True).start()

print("Connected to Coral.")
print("Type y or n exactly as you would in the Serial Monitor.")
print()

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # ---------- Image Capture ----------
        if line == "BEGIN_IMAGE":
            print("\nReceiving image...")
            capturing = True
            pixels = []
            continue

        if line == "END_IMAGE":
            capturing = False

            print(f"Received {len(pixels)} pixels.")

            if len(pixels) == WIDTH * HEIGHT:

                img = np.array(pixels, dtype=np.uint8)
                img = img.reshape((HEIGHT, WIDTH))

                filename = f"capture_{image_count}.png"
                Image.fromarray(img).save(filename)

                print(f"Saved {filename}")

                image_count += 1
            else:
                print("Image size incorrect!")

            continue

        if capturing:
            try:
                pixels.append(int(line))
            except ValueError:
                pass
            continue

        # ---------- Normal Serial Output ----------
        print(line)

    except KeyboardInterrupt:
        break

ser.close()