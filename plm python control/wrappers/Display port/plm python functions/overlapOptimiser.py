import numpy as np
from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
from ampModPhase import amp_mod_phase
from pypylon import pylon
import os
from datetime import datetime
import time
from basler_centroid import baslerCentroid
import cv2
from scipy.interpolate import LinearNDInterpolator
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
import sys
from PyQt5.QtWidgets import QApplication, QInputDialog
import matplotlib.pyplot as plt
from simpleBeamMaker import simple_beam_maker
from scipy.ndimage import zoom
from datetime import datetime
from findGlobalPhaseMinimum2 import find_global_phase_minimum_2

numHolograms = 24
cols, rows = 1358 , 800
images_per_batch = 10

def overlap_optimiser(self, plm, camera):

    plm.pause_ui()
    plm_phase_map = simple_beam_maker(self, 1, 0, 0, 0, 0) # First only switch on Beam A

    plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (rows, cols, numHolograms)).astype(np.float32)
    plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')                        
    plm.bitpack_and_insert_gpu(plm_frame, 2)
    plm.resume_ui()
    plm.set_frame(2)
    time.sleep(0.2)
    plm.play()
    plm.play()

    if camera.IsGrabbing():
        camera.StopGrabbing()
        # Set to software trigger mode
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')
    camera.StartGrabbingMax(images_per_batch)

    # Get image dimensions
    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    # Pre-allocate array: shape [images_per_batch, height, width]
    all_images = np.zeros((images_per_batch, camera_height, camera_width), dtype=np.uint8)

    grab_result = None

    for image_index in range(images_per_batch):
        # Issue software trigger
        camera.TriggerSoftware.Execute()
        # Wait for image
        time.sleep(0.05)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            img = grab_result.Array
            all_images[image_index, :, :] = img
        else:
            print("Failed to grab image", image_index)

        grab_result.Release()

    beamA_centroid_x, beamA_centroid_y = baslerCentroid(np.mean(all_images, axis=0), 3, 5)
    print('Beam A XY centroid = ' + str(beamA_centroid_x) + ' ' + str(beamA_centroid_y) + ' pix')

    # Use Beam A centroid to find 32 by 32 pix ROI, interpolate by X10 and find higher pixel resolution centroid
    roi_size = 64
    half_roi = roi_size // 2
    cx, cy = int(beamA_centroid_x), int(beamA_centroid_y)

    # Make sure ROI is within bounds
    xmin = max(cx - half_roi, 0)
    xmax = min(cx + half_roi, camera_width)
    ymin = max(cy - half_roi, 0)
    ymax = min(cy + half_roi, camera_height)

    roi_A = np.mean(all_images[:, ymin:ymax, xmin:xmax], axis = 0)
    roi_A_zoom = zoom(roi_A, 10, order=3)
    beamA_centroid_zoom_x, beamA_centroid_zoom_y = baslerCentroid(roi_A_zoom, 3, 5)
    print('Zoomed XY centroid for Beam A = ' + str(beamA_centroid_zoom_x) + ' ' + str(beamA_centroid_zoom_y))

    plm.pause_ui()
    plm_phase_map = simple_beam_maker(self, 0, 1, 0, 0, 0) # Now only switch on Beam B

    plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (rows, cols, numHolograms)).astype(np.float32)
    plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')                        
    plm.bitpack_and_insert_gpu(plm_frame, 2)
    plm.resume_ui()
    plm.set_frame(2)
    time.sleep(0.2)
    plm.play()
    plm.play()

    if camera.IsGrabbing():
        camera.StopGrabbing()
    # Set to software trigger mode
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')
    camera.StartGrabbingMax(images_per_batch)

    # Get image dimensions
    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    # Pre-allocate array: shape [images_per_batch, height, width]
    all_images = np.zeros((images_per_batch, camera_height, camera_width), dtype=np.uint8)

    grab_result = None

    for image_index in range(images_per_batch):
        # Issue software trigger
        camera.TriggerSoftware.Execute()
        # Wait for image
        time.sleep(0.05)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            img = grab_result.Array
            all_images[image_index, :, :] = img
        else:
            print("Failed to grab image", image_index)

        grab_result.Release()

    beamB_centroid_x, beamB_centroid_y = baslerCentroid(np.mean(all_images, axis=0), 3, 5)
    print('\nBeam B XY centroid = ' + str(beamB_centroid_x) + ' ' + str(beamB_centroid_y) + ' pix')

    xdiff = beamA_centroid_x - beamB_centroid_x
    ydiff = beamA_centroid_y - beamB_centroid_y
    k = 0.005        # k = Search tuning parameter => Larger values mean faster search, by might start oscillating
    cumulative_tilt_x = 0
    cumulative_tilt_y = 0
    while abs(xdiff) > 5 or abs(ydiff) > 5:
        print(f'\nBeamA - BeamB X diff = {xdiff:.2f} pixels')
        print(f'BeamA - BeamB Y diff = {ydiff:.2f} pixels')

        xtilt_shift = 20*k * xdiff  
        ytilt_shift = k * ydiff

        cumulative_tilt_x += xtilt_shift
        cumulative_tilt_y += ytilt_shift

        print(f'\n\nCumulative X tilt shift of {np.round(cumulative_tilt_x,2)}')
        print(f'Cumulative Y tilt shift of {np.round(cumulative_tilt_y,2)}')

        plm.pause_ui()
        plm_phase_map = simple_beam_maker(self, 0, 1, -1*cumulative_tilt_x,1*cumulative_tilt_y, 0)  
        plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (rows, cols, numHolograms)).astype(np.float32)
        plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')                        
        plm.bitpack_and_insert_gpu(plm_frame, 2)
        plm.resume_ui()
        plm.set_frame(2)
        time.sleep(0.2)
        plm.play()
        plm.play()

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
            else:
                print("Failed to grab image", image_index)

            grab_result.Release()

        beamB_centroid_x, beamB_centroid_y = baslerCentroid(np.mean(all_images, axis=0), 3, 5)
        print('Beam A XY centroid = ' + str(beamA_centroid_x) + ' ' + str(beamA_centroid_y) + ' pix')
        print('Beam B XY centroid = ' + str(beamB_centroid_x) + ' ' + str(beamB_centroid_y) + ' pix')
        xdiff = beamA_centroid_x - beamB_centroid_x
        ydiff = beamA_centroid_y - beamB_centroid_y

    print('\n\nCentroids within 5 pixels, start sub-pixel resolution search')

    roi_B = np.mean(all_images[:, ymin:ymax, xmin:xmax], axis = 0)
    roi_B_zoom = zoom(roi_B, 10, order=3)
    beamB_centroid_zoom_x, beamB_centroid_zoom_y = baslerCentroid(roi_B_zoom, 3, 5)
    print('Zoomed XY centroid for Beam B = ' + str(beamB_centroid_zoom_x) + ' ' + str(beamB_centroid_zoom_y))

    xdiff = beamA_centroid_zoom_x - beamB_centroid_zoom_x
    ydiff = beamA_centroid_zoom_y - beamB_centroid_zoom_y
    k = 0.0005        # k = Search tuning parameter => Larger values mean faster search, but might start oscillating
    cumulative_tilt_x_zoom = 0
    cumulative_tilt_y_zoom = 0

    while abs(xdiff) > 2 or abs(ydiff) > 2:
        print(f'\nBeamA - BeamB X diff = {xdiff:.2f} zoomed pixels')
        print(f'BeamA - BeamB Y diff = {ydiff:.2f} zoomed pixels')

        if abs(xdiff) > 1:
            xtilt_shift = 40*k * xdiff
            cumulative_tilt_x_zoom += xtilt_shift
        else:
            xtilt_shift = 0  

        if abs(ydiff) > 1:
            ytilt_shift = k * ydiff
            cumulative_tilt_y_zoom += ytilt_shift
        else:
            ytilt_shift = 0  

        print(f'\n\nCumulative X tilt shift of {np.round(cumulative_tilt_x_zoom,2)}')
        print(f'Cumulative Y tilt shift of {np.round(cumulative_tilt_y_zoom,2)}')

        plm.pause_ui()
        plm_phase_map = simple_beam_maker(self, 0, 1, -1*(cumulative_tilt_x +cumulative_tilt_x_zoom) , 1*(cumulative_tilt_y + cumulative_tilt_y_zoom), 0)  
        plm_frame = np.broadcast_to(plm_phase_map[:, :, None], (rows, cols, numHolograms)).astype(np.float32)
        plm_frame = np.transpose(plm_frame, (1, 0, 2)).copy(order='F')                        
        plm.bitpack_and_insert_gpu(plm_frame, 2)
        plm.resume_ui()
        plm.set_frame(2)
        time.sleep(0.2)
        plm.play()
        plm.play()

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
            else:
                print("Failed to grab image", image_index)

            grab_result.Release()

        roi_B = np.mean(all_images[:, ymin:ymax, xmin:xmax], axis = 0)
        roi_B_zoom = zoom(roi_B, 10, order=3)
        beamB_centroid_zoom_x, beamB_centroid_zoom_y = baslerCentroid(roi_B_zoom, 3, 5)
        print('Zoomed XY centroid for Beam B = ' + str(beamB_centroid_zoom_x) + ' ' + str(beamB_centroid_zoom_y))
        
        print('Beam A zoom XY centroid = ' + str(beamA_centroid_zoom_x) + ' ' + str(beamA_centroid_zoom_y) + ' pix')
        print('Beam B zoom XY centroid = ' + str(beamB_centroid_zoom_x) + ' ' + str(beamB_centroid_zoom_y) + ' pix')
        xdiff = beamA_centroid_zoom_x - beamB_centroid_zoom_x
        ydiff = beamA_centroid_zoom_y - beamB_centroid_zoom_y

    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    date_folder = os.path.join(os.getcwd(), 'Data', date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Create subfolder with current time inside the date folder
    time_str = now.strftime("%H_%M_%S") + ' overlap phase scanning'
    time_folder = os.path.join(date_folder, time_str)
    os.makedirs(time_folder, exist_ok=True)

    offline_date_folder = os.path.join(r'C:\Users\bs426\OneDrive - University of Exeter\!Work\Work.2025\Lab.2025\110B\Images for offline analysis',date_str)
    os.makedirs(offline_date_folder, exist_ok=True)
    offline_time_folder = os.path.join(offline_date_folder, time_str)
    os.makedirs(offline_time_folder, exist_ok=True)

    roi_sums = []

    new_B_x_tilt = self.user_values[2] + -1*(cumulative_tilt_x +cumulative_tilt_x_zoom)
    new_B_y_tilt = self.user_values[3] + (cumulative_tilt_y +cumulative_tilt_y_zoom)

    print('Beam A x tilt = ' + str(np.round(self.user_values[0],3)))
    print('Beam A y tilt = ' + str(np.round(self.user_values[1], 3)))

    print('Beam B x tilt = ' + str(np.round(new_B_x_tilt,3)))
    print('Beam B y tilt = ' + str(np.round(new_B_y_tilt, 3)))


    roi_slice = (slice(ymin, ymax), slice(xmin, xmax))
    
    zero_phase = find_global_phase_minimum_2(
        self, plm, camera,
        cumulative_tilt_x, cumulative_tilt_y,
        cumulative_tilt_x_zoom, cumulative_tilt_y_zoom,
        rows, cols,
        images_per_batch=10,
        roi_slice=roi_slice
    )

    return new_B_x_tilt , new_B_y_tilt , zero_phase
