import serial
import numpy as np
import threading
from PIL import Image

PORT = "/dev/ttyACM0"
BAUD = 115200

WIDTH = 96
HEIGHT = 96
BYTES_PER_FRAME = WIDTH * HEIGHT

image_count = 0

ser = serial.Serial(PORT, BAUD)

print("Connected to Arduino.")


def keyboard_thread():
    while True:
        try:
            text = input()
            ser.write(text.encode())
        except KeyboardInterrupt:
            break


threading.Thread(target=keyboard_thread, daemon=True).start()


while True:
    try:
        # Wait for BEGIN_PHOTO
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        print(line)

        if line == "BEGIN_PHOTO":

            print("Receiving image...")

            # Read the entire image
            pixels = ser.read(BYTES_PER_FRAME)

            print(f"Received {len(pixels)} bytes.")

            if len(pixels) != BYTES_PER_FRAME:
                print("Image size incorrect!")
                continue

            # Convert bytes to NumPy
            img = np.frombuffer(pixels, dtype=np.uint8)

            # Reshape
            img = img.reshape((HEIGHT, WIDTH))

            # Save
            filename = f"capture_{image_count}.png"
            Image.fromarray(img).save(filename)

            print(f"Saved {filename}")

            image_count += 1

            # Now read END_PHOTO
            line = ser.readline().decode(errors="ignore").strip()

            print("Received:", line)

            if line == "END_PHOTO":
                print("Photo complete.")

    except KeyboardInterrupt:
        break

ser.close()