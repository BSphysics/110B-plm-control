

import numpy as np
import os
import cv2 
from datetime import datetime


def save_super_pixel_images(data, beamName):
    # data = np.load('Beam A images.npy') 
    print('Number of images acquired = ' + str(np.shape(data)))
    # Create 'images' directory if it doesn't exist
    now = datetime.now()

    date_str = now.strftime("%Y_%m_%d")
    date_folder = os.path.join(os.getcwd(), 'Data', date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Create subfolder with current time inside the date folder
    time_str = now.strftime("%H_%M_%S") + beamName
    time_folder = os.path.join(date_folder, time_str)
    os.makedirs(time_folder, exist_ok=True)

    offline_date_folder = os.path.join(r'C:\Users\bs426\OneDrive - University of Exeter\!Work\Work.2025\Lab.2025\110B\Images for offline analysis',date_str)
    os.makedirs(offline_date_folder, exist_ok=True)
    offline_time_folder = os.path.join(offline_date_folder, time_str)
    os.makedirs(offline_time_folder, exist_ok=True)

    # Loop through each image and save it
    for i, img in enumerate(data):
        filename = os.path.join(time_folder, f"image_{i:03d}.png")  
        cv2.imwrite(filename, img)  

        off_line_filename = os.path.join(offline_time_folder, f"image_{i:03d}.png")  
        cv2.imwrite(off_line_filename, img)  
        

    print("All images saved successfully!")

    filename = os.path.join(time_folder, beamName) 
    np.save(filename, data)

    offline_filename = os.path.join(offline_time_folder, beamName) 
    np.save(offline_filename, data)

    return time_folder
