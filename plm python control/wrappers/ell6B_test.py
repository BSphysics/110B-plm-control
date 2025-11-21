import serial
import time

PORT = "COM4"  # change this to your COM port
ser = serial.Serial(PORT, baudrate=9600, timeout=0.5)

def send(cmd):
    ser.write((cmd + "\r").encode("ascii"))
    time.sleep(0.1)
    response = ser.read_all().decode(errors="ignore")
    if response:
        print("Response:", response.strip())
    return response

# Test sequence
send("I")   # query identity
time.sleep(0.5)

send("H")   # home stage
time.sleep(1)

send("F")   # move forward
time.sleep(1)

send("B")   # move backward
time.sleep(1)

send("P")   # get position

ser.close()
