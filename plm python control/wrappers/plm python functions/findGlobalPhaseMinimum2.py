import numpy as np
import time
import os
from simpleBeamMaker import simple_beam_maker
from pypylon import pylon

numHolograms = 24
cols, rows = 1358 , 800

def find_global_phase_minimum_2(
        self, plm, camera,
        cumulative_tilt_x, cumulative_tilt_y,
        cumulative_tilt_x_zoom, cumulative_tilt_y_zoom,
        rows, cols,
        images_per_batch=6,
        roi_slice=None,
        background_width=64,
        coarse_steps=21,
        fine_span=6,
        fine_steps=11,
        save_path=None):

    """
    Deterministic two-stage global phase minimisation.
    """

    ymin, ymax = roi_slice[0].start, roi_slice[0].stop
    xmin, xmax = roi_slice[1].start, roi_slice[1].stop

    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    if camera.IsGrabbing():
        camera.StopGrabbing()
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue('Software')

    def measure_intensity(global_phase_percent):

        # ---- Generate phase map ----
        plm.pause_ui()
        plm_phase_map = simple_beam_maker(
            self, 0.8, 1,
            -1*(cumulative_tilt_x + cumulative_tilt_x_zoom),
            1*(cumulative_tilt_y + cumulative_tilt_y_zoom),
            global_phase_percent
        )

        plm_frame = np.broadcast_to(
            plm_phase_map[:, :, None],
            (rows, cols, numHolograms)
        ).astype(np.float32)

        plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')
        plm.bitpack_and_insert_gpu(plm_frame, 2)
        plm.resume_ui()
        plm.set_frame(2)
        plm.play(); plm.play()

        time.sleep(0.15)  # allow PLM to settle

        # ---- Acquire images ----
        camera.StartGrabbingMax(images_per_batch)

        all_images = np.zeros(
            (images_per_batch, camera_height, camera_width),
            dtype=np.uint8
        )

        for image_index in range(images_per_batch):
            camera.TriggerSoftware.Execute()
            grab_result = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grab_result.GrabSucceeded():
                all_images[image_index] = grab_result.Array
            grab_result.Release()

        # ---- ROI processing ----
        roi = np.mean(all_images[:, ymin:ymax, xmin:xmax], axis=0)

        left_xmin = max(xmin - background_width, 0)
        background_roi = np.mean(
            all_images[:, ymin:ymax, left_xmin:xmin],
            axis=0
        )

        roi_sum = np.sum(roi) - np.sum(background_roi)

        if save_path:
            filename = os.path.join(
                save_path,
                f'Global_phase_{global_phase_percent:.2f}.npy'
            )
            np.save(filename, all_images)

        return roi_sum

    # -------------------------------------------------
    # COARSE SCAN
    # -------------------------------------------------
    print("\nStarting coarse phase scan...")

    coarse_phases = np.linspace(0, 100, coarse_steps)
    coarse_values = []

    for phase in coarse_phases:
        val = measure_intensity(phase)
        coarse_values.append(val)
        print(f"Phase {phase:.2f} -> {val:.1f}")

    coarse_values = np.array(coarse_values)

    best_index = np.argmin(coarse_values)
    coarse_min_phase = coarse_phases[best_index]

    print(f"\nCoarse minimum near {coarse_min_phase:.2f}%")

    # -------------------------------------------------
    # LOCAL REFINEMENT
    # -------------------------------------------------
    print("Refining around coarse minimum...")

    fine_start = max(coarse_min_phase - fine_span/2, 0)
    fine_end   = min(coarse_min_phase + fine_span/2, 100)

    fine_phases = np.linspace(fine_start, fine_end, fine_steps)
    fine_values = []

    for phase in fine_phases:
        val = measure_intensity(phase)
        fine_values.append(val)
        print(f"Refine {phase:.2f} -> {val:.1f}")

    fine_values = np.array(fine_values)

    fine_min_index = np.argmin(fine_values)
    optimal_phase = fine_phases[fine_min_index]

    print(f"\nOptimal global phase (%) = {optimal_phase:.2f}")

    # -------------------------------------------------
# FINAL MEASUREMENT AT OPTIMAL PHASE AND π SHIFT
# -------------------------------------------------
    # Intensity at optimal phase
    I_opt = measure_intensity(optimal_phase)

    # Convert π radians to % scale (π = 50% of 0–100%)
    pi_shift_phase = (optimal_phase + 50) % 100  # wrap around 100%
    I_pi = measure_intensity(pi_shift_phase)

    # Compute ratio
    ratio = (I_opt / I_pi)*100 if I_pi != 0 else np.inf

    print(f"\nBeam extinction percentage = {ratio:.1f}")


    # -------------------------------------------------
    # Restore free streaming
    # -------------------------------------------------
    print('\nSwitching back to free streaming')
    camera.TriggerMode.SetValue('Off')
    camera.OffsetX.SetValue(0)
    camera.OffsetY.SetValue(0)
    camera.Width.SetValue(512)
    camera.Height.SetValue(512)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    return optimal_phase
