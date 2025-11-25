import serial
import time

def slider(self):

    """Toggle the slider between Position A and Position B."""
    # Disable the button to prevent multiple clicks
    self.slider_button.setEnabled(False)

    # -------------------------------
    # FIRST PRESS: initialise position
    # -------------------------------
    if self.slider_position is None:
        print("Initialising slider: forcing to attenuator position..")

        # Send move-to-A command
        init_command = b'0fw\n'   # adjust if reverse for your device

        def initial_move():
            try:
                self.ELLser = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                self.ELLser.reset_input_buffer()
                self.ELLser.flushInput()
                self.ELLser.flushOutput()

                self.ELLser.write(init_command)
                time.sleep(1.5)

            except serial.SerialException as e:
                print(f"Serial error during initialisation: {e}")

            finally:
                if self.ELLser and self.ELLser.is_open:
                    self.ELLser.flushInput()
                    self.ELLser.flushOutput()
                    self.ELLser.close()

        initial_move()

        # Set known state
        self.slider_position = "Attenuator"
        self.update_slider_button_style()

        self.slider_button.setEnabled(True)
        print("Slider initialised to attenuate beam.")
        return   # Done with the initialisation press



    # Determine direction based on current position
    if self.slider_position == "No attenuator":
        command = b'0fw\n'   # move forward to position B
        next_position = "Attenuator"
    else:
        command = b'0bw\n'   # move backward to position A
        next_position = "No attenuator"

    def serial_communication():
        try:
            # Open serial connection
            self.ELLser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.ELLser.reset_input_buffer()
            self.ELLser.flushInput()
            self.ELLser.flushOutput()

            # Optional: query device info
            self.ELLser.write(b'0in\n')
            time.sleep(0.2)
            if self.ELLser.in_waiting > 0:
                print(self.ELLser.readline().decode('ascii'))

            # Send move command
            print(f"Moving slider: {self.slider_position} → {next_position}")
            self.ELLser.write(command)
            time.sleep(1.5)

        except serial.SerialException as e:
            print(f"Serial error: {e}")

        finally:
            if self.ELLser and self.ELLser.is_open:
                self.ELLser.flushInput()
                self.ELLser.flushOutput()
                self.ELLser.close()

    # Run serial communication
    serial_communication()

    # Update stored position
    self.slider_position = next_position

    # Update button colour
    self.update_slider_button_style()

    # Re-enable button
    self.slider_button.setEnabled(True)

    print(f"Slider is now at position {self.slider_position}")
