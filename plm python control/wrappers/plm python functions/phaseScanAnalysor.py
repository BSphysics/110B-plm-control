
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

def baslerCentroid(img, sigma, thresh):
    
    img = gaussian_filter(img, sigma)
    
    mask = img > thresh
    
    rows, cols = np.indices(img.shape)
    
    if np.any(mask):  # Check if any high-intensity pixels exist
    
        centroid_x = int(np.sum(cols * mask * img) / np.sum(img[mask]))
        centroid_y = int(np.sum(rows * mask * img) / np.sum(img[mask]))
        
    else:
        centroid_x = int(48)
        centroid_y = int(48)
        print('These are just placeholder values - beam intensity too low to find proper centroid')
        # return None
       
    return centroid_x, centroid_y


def phase_scan_analysor(imgs):

    img_sum = np.sum(imgs, axis=(1, 2))
    brightest_idx = np.argmax(img_sum)
    brightest_image = imgs[brightest_idx]

    centroid_x, centroid_y = baslerCentroid(
        brightest_image,
        sigma=3,
        thresh=5
    )

    rectangle_half = 10
    height, width = brightest_image.shape

    rect_x_coord = centroid_x - rectangle_half
    rect_y_coord = centroid_y - rectangle_half

    rect_end_x = rect_x_coord + 2 * rectangle_half
    rect_end_y = rect_y_coord + 2 * rectangle_half
   

    imSum=[]
       
    for img in imgs:
        zoomed_image_half_size = 40
        zoomed_img = 1.5*img[centroid_y - zoomed_image_half_size : centroid_y + zoomed_image_half_size ,centroid_x - zoomed_image_half_size : centroid_x + zoomed_image_half_size]
        
        rectangle_half=10
        height, width = zoomed_img.shape
        rect_x_coord = width // 2 - rectangle_half
        rect_y_coord = height // 2 - rectangle_half
        rect_end_x = rect_x_coord + int(2*rectangle_half)
        rect_end_y = rect_y_coord + int(2*rectangle_half)
        
        mask = zoomed_img > 3
        zoomed_img = zoomed_img*mask    
        
        image_sum = float(np.sum(zoomed_img[rect_y_coord : rect_end_y , rect_x_coord : rect_end_x]))
        background_sum = float(np.sum(zoomed_img[rect_y_coord : rect_end_y , rect_x_coord-int(2*rectangle_half) : rect_end_x-int(2*rectangle_half)])) 

        imSum.append(image_sum-background_sum)


    imSum = np.array(imSum)
    imSum=imSum[0 : 792]
    imSum = np.reshape(imSum, (33,24))


    imMean = np.mean(imSum, axis=0)

    #%%


    phase_array = np.arange(0, 100.1, 4.33)

    fine_phases = np.linspace(phase_array[0], phase_array[-1], 1000)  # much finer time grid
    interp_func = interp1d(phase_array, imMean, kind='cubic')
    interpimMean = interp_func(fine_phases)

    plt.figure()
    plt.plot(fine_phases, interpimMean)
    min_index = np.argmin(interpimMean)
    max_index = np.argmax(interpimMean)
    phase_min = np.round(fine_phases[min_index],2)
    print('Use global phase = ' + str(phase_min))

    extinction = np.sqrt(np.min(imMean)/np.max(imMean))*100

    print('E field extinction % = ' + str(np.round(extinction,2)))
                   
