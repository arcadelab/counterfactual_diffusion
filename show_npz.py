
"""show npz file"""

import nibabel as nib
import numpy as np
import torch

# Load the npz file

import os

file_dir = './npz4genunettumour'
for each_npz in os.listdir(file_dir):
    data = np.load(os.path.join(file_dir, each_npz))
    gt = data['gt'].squeeze()
    sample = data['sample'].squeeze()

    # calculate each channels difference by subtracting the gt from the sample
    diff = sample - gt
    padding_mask = data['padding_mask']
    # Show the gt and sample, each 4 channel in one subplot
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    for i in range(4):
        axs[0, i].imshow(gt[i], cmap='gray')
        axs[0, i].set_title('gt channel {}'.format(i))
        axs[0, i].axis('off')
        axs[1, i].imshow(sample[i], cmap='gray')
        axs[1, i].set_title('sample channel {}'.format(i))
        axs[1, i].axis('off')
        axs[2, i].imshow(diff[i], cmap='gray')
        axs[2, i].set_title('diff channel {}'.format(i))
        axs[2, i].axis('off')

    # single plot fot the padding mask
    plt.figure()
    plt.imshow(padding_mask, cmap='gray')
    plt.axis('off')

    plt.show()
