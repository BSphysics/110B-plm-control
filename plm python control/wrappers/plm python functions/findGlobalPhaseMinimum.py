from scipy.optimize import minimize_scalar
import numpy as np
import time
import os
from simpleBeamMaker import simple_beam_maker
from pypylon import pylon

numHolograms = 24
cols, rows = 1358 , 800

def find_global_phase_minimum(self, plm, camera, cumulative_tilt_x, cumulative_tilt_y,
                              cumulative_tilt_x_zoom, cumulative_tilt_y_zoom,
                              rows, cols, images_per_batch=10, roi_slice=None,
                              background_width=64, avg_repeats=3, save_path=None):

    """
    Finds the global phase (%) that minimizes the intensity in the ROI.

    Parameters
    ----------
    images_per_batch : int
        Number of images to average per phase.
    roi_slice : tuple of slices
        The ROI region (y_slice, x_slice).
    background_width : int
        Width of the background ROI for subtraction.
    avg_repeats : int
        Number of repeated measurements per phase to reduce noise.
    save_path : str or None
        Folder to save the acquired images (optional).

    Returns
    -------
    optimal_phase : float
        The global phase (%) that minimizes the ROI intensity.
    """
    
    ymin, ymax = roi_slice[0].start, roi_slice[0].stop
    xmin, xmax = roi_slice[1].start, roi_slice[1].stop

        # Get image dimensions
    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    def measure_intensity(global_phase_percent):
        """Measures background-subtracted ROI sum for a given global phase."""
        roi_sums = []
        for _ in range(avg_repeats):
            print('\n Calculating minimum phase value')
            # Generate phase map and send to PLM
            plm.pause_ui()
            plm_phase_map = simple_beam_maker(
                self, 0.8, 1,
                -1*(cumulative_tilt_x + cumulative_tilt_x_zoom),
                1*(cumulative_tilt_y + cumulative_tilt_y_zoom),
                global_phase_percent
            )
            plm_frame = np.broadcast_to(plm_phase_map[:, :, None], 
                                        (rows, cols, numHolograms)).astype(np.float32)
            plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')
            plm.bitpack_and_insert_gpu(plm_frame, 2)
            plm.resume_ui()
            plm.set_frame(2)
            time.sleep(0.2)
            plm.play(); plm.play()

            # Acquire images
            if camera.IsGrabbing():
                camera.StopGrabbing()
            camera.TriggerSelector.SetValue('FrameStart')
            camera.TriggerMode.SetValue('On')
            camera.TriggerSource.SetValue('Software')
            camera.StartGrabbingMax(images_per_batch)

            all_images = np.zeros((images_per_batch, camera_height, camera_width), dtype=np.uint8)
            for image_index in range(images_per_batch):
                camera.TriggerSoftware.Execute()
                time.sleep(0.05)
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    all_images[image_index, :, :] = grab_result.Array
                grab_result.Release()

            # Compute ROI sum minus background
            roi = np.mean(all_images[:, ymin:ymax, xmin:xmax], axis=0)
            left_xmin = max(xmin - background_width, 0)
            background_roi = np.mean(all_images[:, ymin:ymax, left_xmin:xmin], axis=0)
            roi_sum = np.sum(roi) - np.sum(background_roi)
            roi_sums.append(roi_sum)

            # Optionally save images
            if save_path:
                filename = os.path.join(save_path, f'Global phase (%) = {global_phase_percent:.2f}.npy')
                np.save(filename, all_images)

        # Average repeated measurements
        return np.mean(roi_sums)

    # Use scalar minimization with bounds 0-100%
    res = minimize_scalar(measure_intensity, bounds=(0, 100), method='bounded', options={'xatol':0.1})
    optimal_phase = res.x

    print(f"Optimal global phase (%) = {optimal_phase:.2f}")

        # Reset to free-running / GUI streaming mode
    print('\nSwitching back to free streaming')
    camera.TriggerMode.SetValue('Off')        # Turn off trigger mode
    camera.OffsetX.SetValue(0)                # Optional: reset ROI offset
    camera.OffsetY.SetValue(0)
    camera.Width.SetValue(512)                # Optional: reset resolution to GUI defaults
    camera.Height.SetValue(512)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return optimal_phase
