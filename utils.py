import matplotlib.pyplot as plt
import numpy as np


def plot_image_with_mask(image, mask):
    """
    Plots an image with its corresponding mask.
    """
    plt.imshow(image, cmap='gray')
    masked_image = np.ma.masked_equal(mask, 0)
    plt.imshow(masked_image, cmap='gray')
    plt.show()
