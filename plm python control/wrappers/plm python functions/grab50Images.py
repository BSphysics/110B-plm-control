import numpy as np
from pypylon import pylon
import os
from datetime import datetime
import cv2
from matplotlib import pyplot as plt
import time


def grab_50_images(camera, beamName):

    if camera.IsGrabbing():
        camera.StopGrabbing()

    images_per_batch = 50

    # Set to software trigger mode
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')
    #camera.ExposureTimeAbs.SetValue(1000)

    # Start grabbing manually
    camera.StartGrabbingMax(images_per_batch)
    time.sleep(0.1)

    # Get image dimensions
    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    # Pre-allocate array: shape [images_per_batch, height, width]
    all_images = np.zeros((images_per_batch, camera_height, camera_width), dtype=np.uint8)

    grab_result = None

    for image_index in range(images_per_batch):
        # Issue software trigger
        camera.TriggerSoftware.Execute()
        time.sleep(0.05)

        # Wait for image
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            img = grab_result.Array
            all_images[image_index, :, :] = img
        else:
            print("Failed to grab image", image_index)

        grab_result.Release()

    camera.StopGrabbing()
    print('\nSwitching back to free streaming')
    camera.ExposureTimeAbs.SetValue(200)
    camera.TriggerMode.SetValue("Off")        
    offset_x = 0
    offset_y = 0        
    camera.OffsetX.SetValue(offset_x)
    camera.OffsetY.SetValue(offset_y)
    image_height = 512
    image_width = 512
    camera.Width.SetValue(image_width)
    camera.Height.SetValue(image_height)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    print('Number of images acquired = ' + str(np.shape(all_images)))

    # Create folder with today's date
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    date_folder = os.path.join(os.getcwd(), 'Data', date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Create subfolder with current time inside the date folder
    time_str = now.strftime("%H_%M_%S") + ' grab50'
    time_folder = os.path.join(date_folder, time_str)
    os.makedirs(time_folder, exist_ok=True)

    offline_date_folder = os.path.join(r'C:\Users\bs426\OneDrive - University of Exeter\!Work\Work.2025\Lab.2025\110B\Images for offline analysis',date_str)
    os.makedirs(offline_date_folder, exist_ok=True)
    offline_time_folder = os.path.join(offline_date_folder, time_str)
    os.makedirs(offline_time_folder, exist_ok=True)

    filename = os.path.join(time_folder, beamName) 
    np.save(filename, all_images)


    offline_filename = os.path.join(offline_time_folder, beamName)
    np.save(offline_filename, all_images)

    img = np.mean(all_images, axis=0)
    img = img / np.max(img)
    img = img*255
    #img = img.astype(np.uint8)  # <--- important

    #plt.imsave(os.path.join(time_folder, 'grab50_mean.png'), img, cmap='gray', vmin=0, vmax=255)
    #plt.imsave(os.path.join(offline_time_folder, 'grab50_mean.png'), img, cmap='gray', vmin=0, vmax=255)

    plt.figure()
    plt.imshow(img, cmap = 'gray' , vmin = 0 , vmax = 255)
    plt.axis('off')
    plt.savefig(os.path.join(time_folder, 'grab50_mean.png'))
    plt.savefig(os.path.join(offline_time_folder, 'grab50_mean.png'))
    plt.close('all')

    del all_images

    return time_folder, offline_time_folder

