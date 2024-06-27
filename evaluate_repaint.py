"""evaluate repaint betweenunet and dit"""


import numpy as np
import cv2
from skimage import exposure, filters, morphology
from scipy import ndimage

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def post_process(sub_array):
    sub = sub_array
    adjusted = cv2.convertScaleAbs(sub * 255, alpha=4, beta=-2)
    threshold_value = filters.threshold_otsu(adjusted)
    binary_image = adjusted > threshold_value
    labeled_image, num_features = ndimage.label(binary_image)
    sizes = ndimage.sum(binary_image, labeled_image, range(num_features + 1))
    mask_sizes = sizes >= sizes.max()
    mask_sizes[0] = 0
    return mask_sizes[labeled_image]

def analyze_image(path):
    data = np.load(path)
    sample = data['sample']
    sample_gt = data['gt']
    tumour_gt = data['tumour'] > 0.5

    if sample.ndim == 4:  # Check if there's an extra dimension
        sample = sample.squeeze(0)  # Remove singleton dimension if present
    if sample_gt.ndim == 4:
        sample_gt = sample_gt.squeeze(0)

    tumour_list = []
    for i in range(min(sample.shape[0], 4)):  # Ensure the index is within bounds
        normalized = exposure.match_histograms(sample[i], sample_gt[i])
        diff = sample_gt[i] - normalized
        processed = post_process(diff)
        tumour_list.append(processed)

    weighted_sum = sum(img * 0.25 for img in tumour_list) > 0.25
    closed_image = morphology.remove_small_holes(weighted_sum, area_threshold=64)
    final_image = post_process(closed_image.astype(int))

    dice = dice_coefficient(tumour_gt, final_image)
    iou = iou_score(tumour_gt, final_image)

    return iou, dice

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    folder_path = "/mnt/dit_RePaint_per/dit_exp_results_bigtumour"
    #folder_path = "/mnt/RePaint/unet_repaint_results_bigtumour"
    iou_scores = []
    dice_scores = []
    count = 0
    for each_file in os.listdir(folder_path):
        iou, dice = analyze_image(os.path.join(folder_path, each_file))
        #print("File: ", each_file, " IoU: ", iou, " Dice: ", dice)
        if iou <= 0.6 and dice <= 0.6:
            print("File: ", each_file, " IoU: ", iou, " Dice: ", dice)
            count += 1
        iou_scores.append(iou)
        dice_scores.append(dice)

    #threshold = 0.0
    #iou_scores = np.array(iou_scores)
    #dice_scores = np.array(dice_scores)
    # Filter values
    #iou_scores = iou_scores[iou_scores > threshold]
    #dice_scores = dice_scores[dice_scores > threshold]

    plt.figure(figsize=(10, 5))
    plt.hist(iou_scores, bins=20, alpha=0.7, label='IoU Scores')
    plt.hist(dice_scores, bins=20, alpha=0.7, label='Dice Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of IoU and Dice Scores')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.savefig('/mnt/dit_RePaint_per/iou_dice_distribution_dit_big.png')  # Save the figure
    #plt.savefig('/mnt/RePaint/iou_dice_distribution_unet_big.png')
    plt.close()  # Close the plot

    print(f"Mean IoU: {np.mean(iou_scores)}")
    print(f"Mean Dice: {np.mean(dice_scores)}")
    print(f"Median IoU: {np.median(iou_scores)}")
    print(f"Median Dice: {np.median(dice_scores)}")
    print(f"Max IoU: {np.max(iou_scores)}")
    print(f"Max Dice: {np.max(dice_scores)}")
    print(f"Min IoU: {np.min(iou_scores)}")
    print(f"Min Dice: {np.min(dice_scores)}")
    print(f"Number of samples: {len(iou_scores)}")

# seg_padding0_patient_00548_slice_46.npz,
# seg_padding0_patient_01182_slice_120.npz,
# seg_padding0_patient_01232_slice_92.npz
# seg_padding0_patient_01170_slice_70.npz,
# seg_padding0_patient_00116_slice_66.npz
# seg_padding0_patient_00513_slice_55.npz