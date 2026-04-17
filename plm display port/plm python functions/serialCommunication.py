def serial_communication(self):
        """Function to handle the serial communication."""
        try:
            # Open serial connection
            self.ELLser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.ELLser.reset_input_buffer()
            self.ELLser.flushInput()  # Clear the input buffer
            self.ELLser.flushOutput()  # Clear the output buffer

            # Send the request for information about the first ELL14
            self.ELLser.write(('0in' + '\n').encode('utf-8'))
            time.sleep(0.5)

            # Check if data is available
            if self.ELLser.in_waiting > 0:
                serialString = self.ELLser.readline().decode('ascii')   # Serial message back from ELL14
                print(serialString)

            # Send command to move forward
            self.ELLser.write(b'0fw\n')
            time.sleep(1.5)

            # Send command to move backward
            self.ELLser.write(b'0bw\n')
            time.sleep(1.5)

        except serial.SerialException as e:
            print(f"Serial error: {e}")
        finally:
            if self.ELLser and self.ELLser.is_open:
                self.ELLser.flushInput()
                self.ELLser.flushOutput()
                self.ELLser.close()  # Close the serial connection