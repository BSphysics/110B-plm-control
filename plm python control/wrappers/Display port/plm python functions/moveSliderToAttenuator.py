# slider_utils.py

import time
import serial

def move_slider_to_attenuator(self):
    """
    Moves the slider to the Attenuator position via the '0bw' command.
    Safe to call any time multibeam mode starts.
    """

    init_command = b'0bw\n'

    try:
        ser = serial.Serial(
            port=self.serial_port,
            baudrate=self.baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )

        ser.reset_input_buffer()
        ser.flushInput()
        ser.flushOutput()

        ser.write(init_command)
        time.sleep(1.5)

    except serial.SerialException as e:
        print(f"[SLIDER ERROR] Serial error while moving to attenuator: {e}")

    finally:
        try:
            if ser and ser.is_open:
                ser.flushInput()
                ser.flushOutput()
                ser.close()
        except:
            pass
