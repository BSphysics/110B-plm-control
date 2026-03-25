import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

def find_n_centroids(folder_name, n_spots):

    # images_path = os.path.join(folder_name, "grab50_mean.png")

    ROI_HALF_SIZE = 5
    NROWS, NCOLS = 7, 7

    # img = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)
    npy_path = os.path.join(folder_name, "grab50.npy")
    all_images = np.load(npy_path)
    img = np.mean(all_images, axis=0).astype(np.float32)
    # No normalisation - use raw counts
    # img = img[150:250,300:400].astype(np.float32)
    img = img[140:240,255:355].astype(np.float32)

    img_smooth = gaussian_filter(img, sigma=1)

    coordinates = peak_local_max(
        img_smooth,
        min_distance=5,
        num_peaks=n_spots
    )

    if len(coordinates) != n_spots:
        print(f"Warning: detected {len(coordinates)} peaks")

    coords = np.array(coordinates)

    ys = coords[:,0]
    xs = coords[:,1]

    # ------------------------------------------------
    # Reconstruct full 7×7 grid
    # ------------------------------------------------

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    x_grid = np.linspace(x_min, x_max, NCOLS)
    y_grid = np.linspace(y_min, y_max, NROWS)

    # ------------------------------------------------
    # Assign beam numbers
    # ------------------------------------------------

    beam_numbers = []
    roi_sums = []
    centroids = []

    img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, (y, x) in enumerate(coords):

        col_idx = np.argmin(np.abs(x_grid - x))
        row_idx = np.argmin(np.abs(y_grid - y))

        beam_number = col_idx * NROWS + row_idx + 1   # 1-based numbering

        # Draw the calculated beam number on the image
        cv2.putText(img_color, str(int(beam_number)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        beam_numbers.append(beam_number)
        centroids.append((y, x))

        x = int(round(x))
        y = int(round(y))

        x1 = max(0, x - ROI_HALF_SIZE)
        x2 = min(img.shape[1]-1, x + ROI_HALF_SIZE)
        y1 = max(0, y - ROI_HALF_SIZE)
        y2 = min(img.shape[0]-1, y + ROI_HALF_SIZE)

        roi_sum = np.sum(img[y1:y2+1, x1:x2+1])
        roi_sums.append(roi_sum)

        cv2.rectangle(img_color,(x1,y1),(x2,y2),(0,255,0),1)

        if 0 <= y < img_color.shape[0] and 0 <= x < img_color.shape[1]:
            img_color[y,x] = (255,0,0)

    # ------------------------------------------------
    # Save diagnostic image
    # ------------------------------------------------

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Detected Spots")
    plt.axis('off')

    save_path = os.path.join(folder_name,"spot_ROIs.png")
    plt.savefig(save_path)
    plt.close(fig)

    def contrast_norm(roi_sums):
        """Compute normalized contrast from a list of ROI sums."""
        if not roi_sums:  # handle empty list
            return 0.0
        max_val = max(roi_sums)
        min_val = min(roi_sums)
        mean_val = sum(roi_sums) / len(roi_sums)
        if mean_val == 0:
            return 0.0
        return (max_val - min_val) / mean_val

    cn = contrast_norm(roi_sums)
    print(f"Brightest - dimmest spot (normalised to the mean) = {cn:.3f}")

    # print("Detected spot positions:")
    # for bid, (y, x) in zip(beam_numbers, centroids):
    #     print(f"  Spot {bid}: y={y:.1f}, x={x:.1f}")

    return beam_numbers, centroids, roi_sums