# camera_utils.py

import time
from pypylon import pylon

def enable_hardware_trigger(self):
    """
    Configure the Basler camera for hardware-triggered acquisition on Line 3.
    Sets image size, offsets, exposure, and trigger mode.
    """

    print("Hardware triggering enabled")

    # Stop software grabbing if active
    if self.camera.IsGrabbing():
        self.camera.StopGrabbing()

    print("Switching to hardware-triggered acquisition...")

    # Number of images to collect
    self.countOfImagesToGrab = 1800
    self.image_height = 128
    self.image_width = 128
    self.offset_x = 256
    self.offset_y = 160

        # ---- Image Shape ----

    self.camera.Width.SetValue(self.image_width)
    self.camera.Height.SetValue(self.image_height)
    self.camera.OffsetX.SetValue(self.offset_x)
    self.camera.OffsetY.SetValue(self.offset_y)

    # ---- Trigger Configuration ----
    # Disable trigger first
    self.camera.TriggerMode.SetValue("Off")

    # Select GPIO line for trigger input
    self.camera.LineSelector.SetValue("Line3")
    self.camera.LineMode.SetValue("Input")

    # Hardware trigger on Line3, FrameStart on rising edge
    self.camera.TriggerSelector.SetValue("FrameStart")
    self.camera.TriggerSource.SetValue("Line3")
    self.camera.TriggerActivation.SetValue("RisingEdge")
    self.camera.TriggerMode.SetValue("On")

    # ---- Exposure ----
    self.camera.ExposureTimeAbs.SetValue(800)

    # Start waiting for triggers
    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    time.sleep(0.05)

    print(" Waiting for Acquisition Start trigger on Line 3...")
