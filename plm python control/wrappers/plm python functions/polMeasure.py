# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:38:46 2023

@author: BES (b.sherlock@exeter.ac.uk)

"""
import numpy as np
import serial
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pypylon import pylon
import os
from datetime import datetime
scriptDir = os.getcwd()
import sys
sys.path.append(os.path.join(scriptDir,"plm python functions" ))
sys.path.append(os.path.join(scriptDir,"basler python functions" ))
from scipy.ndimage import gaussian_filter

from basler_centroid import baslerCentroid

def pol_measure(camera, global_amplitudes, beamName):

    if camera.IsGrabbing():
        camera.StopGrabbing()
    degrees = np.pi/180
    serialString = ""  # declare a string variable

    ELLser = serial.Serial(         # Open a serial connection to the ELL14. Note you can use Windows device manager to move the USB serial adapter to a different COM port if you need
        port='COM5',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    ELLser.reset_input_buffer()
    ELLser.flushInput()             # Adding these flushes massively helped with the Serial port sending the wrong values and messing up the whole sequence
    ELLser.flushOutput()            # Adding these flushes massively helped with the Serial port sending the wrong values and messing up the whole sequence

    def degreestoHex(deg):          # Quick fn to convert degrees of rotation into the number of pulses needed to actuate this rotation (number in hexadecimal) 
        # first convert degrees to pulses
        pulses = int(deg/360*143360)    # # 143360 is number of pulses needed for 360 degrees of rotation on the ELL14
        
        # convert pulses into hex
        hexPulses = hex(pulses).upper()  #  Hex characters have to be capitals
        return hexPulses[2:]

    jogStepDeg = 20
    jogStepSize = str(degreestoHex(jogStepDeg))   #Set jog step size (in degrees) here

    if len(jogStepSize)<4:
        jogStepSize = jogStepSize.zfill(4)
        
    def serialtoDeg(serialString): #converts hex numbers from stage into degrees
        pos = round((int(serialString.strip()[3:],16)/143360*360),2)        # 143360 is number of pulses needed for 360 degrees of rotation on the ELL14
        return pos

    ELLser.write(('1in' + '\n').encode('utf-8'))    # request information about the first ELL14
    time.sleep(0.2)
    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')   # Serial message back from ELL14            
        print(serialString)

    writeString = '1sj0000'+ str(jogStepSize)      # Set jog step size for ELL14
    ELLser.write((writeString).encode('utf-8'))                            
    time.sleep(0.2)
    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')                
        print(serialString)
        
    ELLser.write(("1gj" + "\n").encode('utf-8'))    # Check that jog step size has been set correctly           
    time.sleep(0.1)
    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')                   
        print('Jog step size = ' + str(round((int(serialString.strip()[3:],16)/143360*360),2)) + ' deg\n')


    ELLser.write(('1ho' + '\n').encode('utf-8'))    # Home ELL14
    time.sleep(1)
    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')                
       
       
    ELLser.write(('1gp' + "\n").encode('utf-8'))    # Check stage position (to make sure homing worked properly)       
    time.sleep(1)
    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')
        pos0 = serialtoDeg(serialString)
        
        if pos0 > 143360:
            pos0 = 0
        print('Starting position of linear polariser = ' + str(pos0) + ' deg' + '\n')

    print('Loop starts \n\n')
    time.sleep(0.1)

    polariserAngles = np.arange(0,200,jogStepDeg)

    ################################################################################

    num_batches = len(polariserAngles)
    images_per_batch = 50

    # Set to software trigger mode
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')
    camera.ExposureTimeAbs.SetValue(180)

    # Start grabbing manually
    camera.StartGrabbingMax(num_batches * images_per_batch)

    # Get image dimensions
    camera_width = camera.Width.Value
    camera_height = camera.Height.Value

    # Pre-allocate array: shape [num_batches, images_per_batch, height, width]
    all_images = np.zeros((num_batches, images_per_batch, camera_height, camera_width), dtype=np.uint8)

    grab_result = None

    idx=0

    mean_images = np.zeros((num_batches, camera_height, camera_width), dtype=np.float32)
    for i in tqdm(polariserAngles):    # In each iteration,  ELL14 takes a regular sized jog step
        print('\nTarget position of linear polariser = ' + str(polariserAngles[idx]) + ' deg')  
        print('\nCurrent position of linear polariser = ' + str(pos0) + ' deg' + '\n') 
        # print(np.isclose(polariserAngles[idx], pos0))

        if not np.isclose(polariserAngles[idx], pos0, atol=0.2):
            break


        #-----------ACQURE IMAGES HERE ----------------
        start_time = time.time()

        for image_index in range(images_per_batch):
            # Issue software trigger
            camera.TriggerSoftware.Execute()

            # Wait for image
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                img = grab_result.Array
                all_images[idx, image_index, :, :] = img
            else:
                print("Failed to grab image", image_index)

            grab_result.Release()

        end_time = time.time()
        duration = end_time - start_time
        print(f"Grabbed {images_per_batch} images in {duration:.2f} seconds")
        #----------------------------------------------
            
        ELLser.write(('1fw' + '\r\n').encode('utf-8'))    #Jog step
        time.sleep(1)
        if(ELLser.in_waiting > 0):
            serialString = ELLser.readline().decode('ascii')                
            pos0 = round(serialtoDeg(serialString))
            
            if pos0 > 143360:
                pos0 = 0


        idx += 1      
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

    cmd = "1ma" + "0000" + degreestoHex(83) + "\r\n"
    print(cmd)
    ELLser.write(cmd.encode("utf-8"))

    time.sleep(1)

    if(ELLser.in_waiting > 0):
        serialString = ELLser.readline().decode('ascii')                
    ELLser.close()

    print('Number of images acquired = ' + str(np.shape(all_images)))
    mean_images = np.mean(all_images, axis=1)
    # Create 'images' directory if it doesn't exist
    now = datetime.now()
    timestamp_str = now.strftime("%Y_%m_%d___%H_%M_%S") +' Pol measurement '
    s1 = ' BeamA=' + str(global_amplitudes[0]) + ' BeamB=' + str(global_amplitudes[1])

    date_str = now.strftime("%Y_%m_%d")
    date_folder = os.path.join(os.getcwd(), 'Data', date_str)
    os.makedirs(date_folder, exist_ok=True)

    # Create subfolder with current time inside the date folder
    time_str = now.strftime("%H_%M_%S") + beamName + s1
    time_folder = os.path.join(date_folder, time_str)
    os.makedirs(time_folder, exist_ok=True)

    offline_date_folder = os.path.join(r'C:\Users\bs426\OneDrive - University of Exeter\!Work\Work.2025\Lab.2025\110B\Images for offline analysis',date_str)
    os.makedirs(offline_date_folder, exist_ok=True)
    offline_time_folder = os.path.join(offline_date_folder, time_str)
    os.makedirs(offline_time_folder, exist_ok=True)

    filename = os.path.join(time_folder, beamName + s1) 
    np.save(filename, mean_images)

    offline_filename = os.path.join(offline_time_folder, beamName + ' Beam A amplitude = ' + str(global_amplitudes[0]) + ' Beam B amplitude = ' + str(global_amplitudes[1]))
    np.save(offline_filename, mean_images)

    del all_images


    # degrees = np.pi/180
    # polariserAngles = np.arange(0,200,20)

    # imgs = mean_images

    # powers=[]

    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
    # plt.subplots_adjust(hspace=0.5)
    # fig.suptitle("Zoomed images", fontsize=8, y=0.95)

    # brightest_index = np.argmax(imgs.sum(axis=(1, 2)))  
    # brightest_image = imgs[brightest_index]

    # centroid_x, centroid_y = baslerCentroid(brightest_image, 3, 5)
    # for img, ax in zip(imgs, axs.ravel()):
    #     #centroid_x, centroid_y = baslerCentroid(img, 3, 5)
    #     zoomed_image_half_size = 40
    #     zoomed_img = img[centroid_y - zoomed_image_half_size : centroid_y + zoomed_image_half_size ,centroid_x - zoomed_image_half_size : centroid_x + zoomed_image_half_size]
        
    #     rectangle_half=10
    #     height, width = zoomed_img.shape
    #     rect_x_coord = width // 2 - rectangle_half
    #     rect_y_coord = height // 2 - rectangle_half
    #     rect_end_x = rect_x_coord + int(2*rectangle_half)
    #     rect_end_y = rect_y_coord + int(2*rectangle_half)
        
    #     mask = zoomed_img > 10
    #     zoomed_img = zoomed_img*mask    
    #     # zoomed_img = gaussian_filter(zoomed_img,3)
        
    #     image_sum = float(np.sum(zoomed_img[rect_y_coord : rect_end_y , rect_x_coord : rect_end_x]))
    #     background_sum = float(np.sum(zoomed_img[rect_y_coord : rect_end_y , rect_x_coord-int(2*rectangle_half) : rect_end_x-int(2*rectangle_half)])) 
    #     powers.append(image_sum-background_sum)

       
    #     ax.imshow(zoomed_img, cmap = 'gray', vmin = 0, vmax=255)
    #     ax.axis('off')

    #     figName = os.path.join(time_folder, 'fig1.png') 
    #     plt.savefig(figName, dpi='figure')

    #     figName = os.path.join(offline_time_folder, 'fig1.png') 
    #     plt.savefig(figName, dpi='figure')

  
    # from matplotlib import patches

    # fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,6))

    # ax1.plot(polariserAngles,powers,'ro')
    # ax1.set_xlabel('Polariser Angle (deg)')

    # from scipy.optimize import curve_fit

    # def model_f(theta,p1,p2,p3,p4):

    #   return (p1*np.cos(theta*degrees-p3))**2 + (p2*np.sin(theta*degrees-p3))**2 +p4

    # try: 
    #     popt, pcov = curve_fit(model_f, polariserAngles, powers, bounds = ([0,0,0,0] , [np.max(powers)*1.5 , np.min(powers)*1.2+1e-5, 2*np.pi, 0.01]))

    #     Ex, Ey, alpha, offset = popt
    #     fittingAngles = np.arange(0,180,1)

    #     ax1.plot(fittingAngles,model_f(fittingAngles, Ex, Ey, alpha, offset),'--b')
    #     ax1.set_ylim(0,np.max(powers)*1.1)
    #     print('\nEllipse semi major axis angle = ' + str(np.round(alpha*180/np.pi,1)) + ' degrees \n')

    #     e1 = patches.Ellipse((0, 0), Ex/2, Ey/2,
    #                      angle=alpha*180/np.pi, linewidth=4,
    #                      fill=False, edgecolor='red', linestyle=(0, (5, 5)), zorder=1)

    #     e2 = patches.Ellipse((0, 0), Ex/2, Ey/2,
    #                          angle=alpha*180/np.pi, linewidth=4,
    #                          fill=False, edgecolor='yellow', linestyle=(5, (5, 5)), zorder=2)

    #     ax2.add_patch(e1)
    #     ax2.add_patch(e2)
    #     ax2.set_xlim([-50,50])
    #     ax2.set_ylim([-50,50])
    #     # ax2.axis('square')
    #     ax2.axis('off')
    #     plt.suptitle('Ellipse semi major axis = ' + str(np.round(alpha*180/np.pi)) + ' deg, ' + 'Emax = ' + str(np.round(Ex,2)) + ', Emin = ' + str(np.round(Ey,2) ))
        
    #     figName = os.path.join(time_folder, 'fig2.png') 
    #     plt.savefig(figName, dpi='figure')

    #     figName = os.path.join(offline_time_folder, 'fig2.png') 
    #     plt.savefig(figName, dpi='figure')

    #     print('Fitted Emax = ' + str(np.round(Ex,3)))
    #     print('Fitted Emin = ' + str(np.round(Ey,3)))
    #     print('Ellipticity = ' + str(np.round(Ey / Ex,3)))

    # except RuntimeError as e:
    #     print('fit failed', e)
    #     ax1.set_title('Fit failed')
    # plt.close('all')


        