# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:51:32 2025

@author: bs426
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:35:07 2025

@author: bs426
"""

from pypylon import pylon
from pypylon import genicam

import numpy as np
import os
scriptDir = os.getcwd()
import sys
# sys.path.append(os.path.join(scriptDir,"!functions" ))
import matplotlib.pyplot as plt
import cv2
import keyboard
import matplotlib.patches as patches
import collections

from camera_config import camConfig

# plt.close('all')

def basler_live_mode():
    fitGaussians = False
    
    #------------ ----------------------Open the camera
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    if len(devices) == 0:
        print("No camera found!")
        exit(1)
    
    camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    camera.Open()
    print("Found Basler ", camera.GetDeviceInfo().GetModelName(), 'camera')
    
    exposureTime = 1000
    camConfig(camera, camera.Height.Max, camera.Width.Max, 0, 0, exposureTime)
    
    from basler_centroid import baslerCentroid
    offset_x, offset_y, centroid_x, centroid_y, zoomed_img = baslerCentroid(camera)
    
    from basler_fitting import baslerFitting
    if fitGaussians == True:
        baslerFitting(zoomed_img)
        while True:
                # Wait until the space bar is pressed so that the fitting plots can be viewed
                print('\nPress space bar to continue')
                if keyboard.is_pressed('space'):
                    print("Space bar pressed, moving to live mode...") 
                    break
    
    #----------------------------------Config ROI parameters
    image_height = 128
    image_width = 128
    exposureTime = 100
    #-------------------------------------------------------
    
    plt.close('all')
    
    camConfig(camera, image_height, image_width, int(offset_x-image_width/2),int(offset_y - image_height/2), exposureTime)
    
    plt.ion()
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.set_axis_off()
    xdata = collections.deque(maxlen=200)  # Store only last 200 points
    ydata = collections.deque(maxlen=200)
    line, = ax2.plot([], [], 'b-')  
     
    ax2.set_xlim(0, 200)
    ax2.set_ylim(10, 2e4) 
    ax2.set_yscale('log')
    img = np.zeros((image_height,image_width))
    im = ax1.imshow(img, cmap='gray', vmin=0, vmax=255)  
    delta = 0 
    roiRect = patches.Rectangle((int(centroid_x)-delta, int(centroid_y)-delta), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
    backgroundRect = patches.Rectangle((int(centroid_x)-delta-31, int(centroid_y)-delta), 30, 30, linewidth=2, edgecolor='b', facecolor='none')
    
    ax1.add_patch(roiRect)
    ax1.add_patch(backgroundRect) 
    
    # Start grabbing images
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("\nPress Space to stop acquisition...")
    t=0
    pixSum=np.zeros(200)
    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
     
            if grab_result.GrabSucceeded():
                
                img = grab_result.Array
                im.set_data(img)
                
                image_sum = float(np.sum(img[int(centroid_y - delta) : int(centroid_y-delta+30) , int(centroid_x - delta) : int(centroid_x-delta+30)]))
                background_sum = float(np.sum(img[int(centroid_y - delta) : int(centroid_y-delta+30) , int(centroid_x - delta-31) : int(centroid_x-delta-1)]))
                xdata.append(t)
                ydata.append(image_sum-background_sum)
                line.set_xdata(range(len(xdata)))
                line.set_ydata(ydata)
    
                ax2.relim()
                ax2.autoscale_view()
    
                fig.canvas.draw()
                fig.canvas.flush_events()
                t+=1
                
                pixSum=np.array(ydata)
                ax2.set_title(str(np.round(np.mean(pixSum[-10:-1]))), fontsize = 16, fontweight='bold' )
                # plt.draw()
                # plt.pause(0.01)
                
    
            grab_result.Release()
    
            if keyboard.is_pressed("space"):
                print("Space bar detected. Stopping acquisition...")
                break
    finally:
        # Stop acquisition and close the camera
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        plt.close(fig)
        print("Acquisition stopped.")
    
    # plt.close('all')