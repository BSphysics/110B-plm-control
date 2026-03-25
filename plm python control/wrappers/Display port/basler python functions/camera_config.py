# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:28:45 2025

@author: bs426
"""

from pypylon import pylon

def camConfig(camera, image_height, image_width, offset_x, offset_y, exposureTime):
    
    camera.OffsetX.SetValue(0)  # Reset horizontal offset
    camera.OffsetY.SetValue(0)  # Reset vertical offset
# Check node availability

    # for node_name in ["Width", "Height", "AcquisitionFrameRateEnable"]:
        # node = camera.GetNodeMap().GetNode(node_name)
        #print(f"{node_name} Exists: {node is not None}")
        
    if camera.GetNodeMap().GetNode("Width") is not None:
        camera.Width.SetValue(image_width)
        #print('\nROI width = ' + str(camera.Width.GetValue()) , 'pixels')
    else:
        print("Warning: 'Width' parameter not found.")
    
    if camera.GetNodeMap().GetNode("Height") is not None:
        camera.Height.SetValue(image_height)
        #print('ROI height = ' + str(camera.Height.GetValue()) , 'pixels')
        # print(camera.Height.GetValue())
    else:
        print("Warning: 'Height' parameter not found.")
    
    if camera.GetNodeMap().GetNode("AcquisitionFrameRateEnable") is not None:
        camera.AcquisitionFrameRateEnable.SetValue(False)
    else:
        print("Warning: 'AcquisitionFrameRateEnable' not found.")
        
    camera.AcquisitionMode.SetValue('Continuous')  # Ensure the camera is in continuous mode
    #camera.AcquisitionFrameRateAbs.SetValue(1000)
    
    camera.OffsetX.SetValue(offset_x)
    # print('\noffset in x = ' + str(camera.OffsetX.GetValue()) , 'pixels')
    
    camera.OffsetY.SetValue(offset_y)
    # print('offset in y = ' + str(camera.OffsetY.GetValue()) , 'pixels')
    
    # Configure External Trigger
    if camera.GetNodeMap().GetNode("TriggerMode") is not None:
    	camera.TriggerMode.SetValue("On")  # Enable trigger mode
    else:
        print("Warning: Problem with TriggerMode")
            
    if camera.GetNodeMap().GetNode("TriggerSource") is not None:
    	camera.TriggerSource.SetValue("Line1")  # External trigger on Line1
    else:
        print("Warning: Problem with TriggerSource")
    
    if camera.GetNodeMap().GetNode("TriggerSelector") is not None:
    	camera.TriggerSelector.SetValue("FrameStart")  # Trigger on frame start
    else:
        print("Warning: Problem with TriggerSelector")
           
    if camera.GetNodeMap().GetNode("TriggerActivation") is not None:
    	camera.TriggerActivation.SetValue("RisingEdge")  # Trigger on rising edge
    else:
        print("Warning: Problem with TriggerActivation")
    
    if camera.GetNodeMap().GetNode("ExposureTimeAbs") is not None:
        camera.ExposureTimeAbs.SetValue(exposureTime)  # set exposure time
        print('\nExposure time = ' + str(camera.ExposureTimeAbs.GetValue()) + 'us')
    else:
        print("Warning: Problem with setting Exposure Time")
    
