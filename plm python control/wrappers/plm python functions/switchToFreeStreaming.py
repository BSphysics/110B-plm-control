def switch_to_free_streaming(self):
    """Return camera + SLM to free-running live streaming mode."""
    print('\nSwitching back to free streaming')

    # Camera: free run, full frame
    self.camera.TriggerMode.SetValue("Off")

    self.camera.OffsetX.SetValue(0)
    self.camera.OffsetY.SetValue(0)

    self.camera.Width.SetValue(512)
    self.camera.Height.SetValue(512)

    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # SLM / PLM: live display
    plm.set_frame(0)
    time.sleep(0.5)
    plm.play()
    plm.play()
