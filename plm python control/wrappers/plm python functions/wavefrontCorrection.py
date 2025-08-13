
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import cv2
from scipy.stats import kurtosis
from scipy.ndimage import gaussian_filter


def wavefront_correction(all_images, saveFolder, beamName):
    print('\nCalculating phase correction pattern...')
    superPixelSize = 64

    xSuperPixels = np.ceil(1358/superPixelSize)
    ySuperPixels = np.ceil(800/superPixelSize)

    referenceSuperPixel_yPosition = 6
    referenceSuperPixel_xPosition = 10

    # use Kurtosis to identify the ON/OFF/ON/OFF sequence
    kurt = np.array([kurtosis(img.flatten()) for img in all_images[0:200]])
    threshold = 0.6 * np.max(kurt)
    high_kurt_indices = np.where(kurt > threshold)[0]
    last_index = high_kurt_indices[-1]

    print('First image used in sequence = ' + str(last_index+2))

    spScan = all_images[last_index+2:] # array of images that starts when SP scan starts

    div_by_5_indices = np.arange(0, len(spScan), 5)    # get only the intensity images 
    int_imgs = spScan[div_by_5_indices]                # get only the intensity images 

    not_div_by_5_indices = np.setdiff1d(np.arange(len(spScan)), div_by_5_indices)  # get only phase step images
    phase_step_imgs = spScan[not_div_by_5_indices]                                 # get only phase step images

    # Number of slices to visualize
    num_slices = 20


    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows, 5 columns
    for i, ax in enumerate(axes.flat):
        ax.imshow(phase_step_imgs[i+100], cmap='gray')  
        ax.set_title(f"Slice {i}")
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig(saveFolder + r'\Phase stepping images.png')
    # plt.pause(0.1)
    plt.close('all')


    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows, 5 columns
    for i, ax in enumerate(axes.flat):
        ax.imshow(int_imgs[i+50], cmap='gray')  
        ax.set_title(f"Slice {i}")
        ax.axis('off')  

    plt.tight_layout()
    plt.savefig(saveFolder + r'\Intensity images.png')
    # plt.pause(0.1)
    plt.close('all')

    
    int_img_sum = np.sum(int_imgs,axis=(1,2))
    brightest_idx = np.argmax(int_img_sum)

    last_high_kurt_image = gaussian_filter(all_images[last_index],3)
    mask = last_high_kurt_image > 5
    # Compute the centroid

    rows, cols = np.indices(last_high_kurt_image.shape)

    centroid_x = np.sum(cols * last_high_kurt_image * mask) / np.sum(last_high_kurt_image[mask])
    centroid_y = np.sum(rows * last_high_kurt_image * mask) / np.sum(last_high_kurt_image[mask])


    # Plot Centroid and save image just in case
    plt.figure()
    plt.imshow(last_high_kurt_image, cmap='gray')  # Display the image in grayscale
    # plt.colorbar()  # Add a colorbar
    plt.scatter(centroid_x, centroid_y, color='red', s=5, label="Centroid")  # Red dot at centroid
    plt.legend()  # Show the label
    plt.title("Centroid test")
    # plt.show()
    plt.savefig(saveFolder + r'\Centroid test.png')
    # plt.pause(0.1)
    plt.close('all')

    ####-------------------------------------------Intensity array 

    # Define the half window size for intensity measurement  
    window_size = 10
    # Find the bounding box for the window around the centroid
    x_min = max(int(centroid_x) - window_size, 0)  # Ensure within array bounds
    x_max = min(int(centroid_x) + window_size + 1, last_high_kurt_image.shape[1])  # Ensure within array bounds
    y_min = max(int(centroid_y) - window_size, 0)
    y_max = min(int(centroid_y) + window_size + 1, last_high_kurt_image.shape[0])

    # Extract the sub-array and sum its values
    int_windows = int_imgs[:,y_min:y_max, x_min:x_max]
    spIntensities1D = np.sum(int_windows, axis=(1,2))

    # for a 100 by 100 super pixel scan we should only have 112 values, therefore slice off any remaining
    totalNumberofSuperPixels = int(xSuperPixels * ySuperPixels)

    spIntensities1D = spIntensities1D[0:totalNumberofSuperPixels]
  
    spIntensities = spIntensities1D.reshape(int(ySuperPixels),int(xSuperPixels))

    #%%
    ####-------------------------------------------Phase array

    all_phases = phase_step_imgs[:,int(np.round(centroid_y)),int(np.round(centroid_x))] # note I've checked and these indicies are the right way round
    all_phases = all_phases[0:4*(totalNumberofSuperPixels)]

    # Create an empty list to store the FFT results
    fft_results = []

    # Loop through the array in chunks of 4 elements
    numberofPhaseSteps = 4
    for i in range(0, len(all_phases), numberofPhaseSteps):
        chunk = all_phases[i:i + numberofPhaseSteps]  # Select the next numberofPhaseSteps elements
        fft_result = np.fft.fft(chunk)  # Perform FFT on the chunk
        fft_results.append(fft_result)  # Store the result

    
    # Convert the list of FFT results into a 2D array
    fft_results_array = np.array(fft_results)

    phase_array = np.angle(fft_results_array[:,1]).reshape(int(ySuperPixels),int(xSuperPixels))

    phase_array[referenceSuperPixel_yPosition,referenceSuperPixel_xPosition+1]=0

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))  

    axes[0].imshow(spIntensities, cmap='gray')  
    axes[0].set_title('Intensity map')
    axes[0].axis('off')  # Hide axes

    cmap = axes[1].imshow(phase_array, cmap='jet') 

    # Use make_axes_locatable to ensure colorbar matches plot height
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size & padding

    # Add colorbar
    cbar = fig.colorbar(cmap, cax=cax)
    cbar.set_label("Phase (Rads)")
     
    axes[1].set_title('Phase map')
    axes[1].axis('off')  # Hide axes

    plt.savefig(saveFolder + r'\amplitude and phase maps.png')
    plt.close('all')

    PLM_phase_array = np.kron(phase_array, np.ones((superPixelSize , superPixelSize),dtype=phase_array.dtype))
    PLM_phase_array = PLM_phase_array[:800,:1358]
    np.savetxt(saveFolder + r'\_' + beamName + 'correction_phase_array.csv', PLM_phase_array[:800,:1358], delimiter = ',', fmt='%.8f')



