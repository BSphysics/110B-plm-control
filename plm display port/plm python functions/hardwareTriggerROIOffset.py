
def hardware_trigger_roi_offset(centroid_x, centroid_y, 
                       roi_width=112, roi_height=90,
                       sensor_width=640, sensor_height=480):
    """
    Compute camera ROI offsets so the ROI is centred on the beam centroid.

    Parameters
    ----------
    centroid_x, centroid_y : float
        Beam centroid position in full-sensor pixel coordinates (from your
        existing centroid-finding function).
    roi_width, roi_height : int
        Fast-acquisition ROI dimensions (fixed at 112 x 90).
    sensor_width, sensor_height : int
        Full CCD size (512 x 512).

    Returns
    -------
    offset_x, offset_y : int
        Values to pass to camera.OffsetX / camera.OffsetY.
    """
    offset_x = int(round(centroid_x - roi_width  / 2))
    offset_y = int(round(centroid_y - roi_height / 2))

    # Clamp so the ROI never exceeds the sensor boundary
    offset_x = max(0, min(offset_x, sensor_width  - roi_width))
    increment = 16   
    offset_x = (offset_x // increment) * increment

    offset_y = max(0, min(offset_y, sensor_height - roi_height))

    return offset_x, offset_y