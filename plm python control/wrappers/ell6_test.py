import serial
import time
import sys

serialString = ""  # declare a string variable

ELLser = serial.Serial(         # Open a serial connection to the ELL14. Note you can use Windows device manager to move the USB serial adapter to a different COM port if you need
    port='COM4',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
ELLser.reset_input_buffer()
ELLser.flushInput()             # Adding these flushes massively helped with the Serial port sending the wrong values and messing up the whole sequence
ELLser.flushOutput() 

ELLser.write(('0in' + '\n').encode('utf-8'))    # request information about the first ELL14
time.sleep(0.5)
if(ELLser.in_waiting > 0):
    serialString = ELLser.readline().decode('ascii')   # Serial message back from ELL14            
    print(serialString)
    

ELLser.write(b'0fw\n')
time.sleep(1.5)

# Jog backward
ELLser.write(b'0bw\n')
time.sleep(1.5)
    
ELLser.flushInput()     # Adding these flushes massively helped with the Serial port sending the wrong values and messing up the whole sequence
ELLser.flushOutput() 

ELLser.close()

