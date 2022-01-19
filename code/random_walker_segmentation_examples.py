import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import random_walker


def show_example_acdc():
    h5f = h5py.File("../imgs/patient001_frame01_slice_5.h5", "r")
    data = h5f["image"][:]
    seed = h5f["scribble"][:]

    # in the seed array: 0 means background, 1 to 3 mean class 1 to 3, 4 means: unknown region
    markers = np.ones_like(seed)
    markers[seed == 4] = 0
    markers[seed == 0] = 1
    markers[seed == 1] = 2
    markers[seed == 2] = 3
    markers[seed == 3] = 4

    sigma = 0.35

    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                             out_range=(-1, 1))

    # Run random walker algorithm
    segmentation = random_walker(data, markers, beta=100, mode='bf')
    labels = segmentation - 1
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('ACDC data')
    ax2.imshow(markers, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Scribbles')
    ax3.imshow(labels, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentation')

    fig.tight_layout()
    plt.savefig("../imgs/acdc_randomwalker_segmentation.png")
    plt.show()
    return "End"


def show_example_protate():
    h5f = h5py.File("../imgs/patient004_slice_8.h5", "r")
    data = h5f["image"][:]
    seed = h5f["scribble"][:]
    # in the seed array: 0 means background, 1 to 2 mean class 1 to 2, 4 means: unknown region
    markers = np.ones_like(seed)
    markers[seed == 4] = 0
    markers[seed == 0] = 1
    markers[seed == 1] = 2
    markers[seed == 2] = 3

    sigma = 0.35
    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                             out_range=(-1, 1))

    # Run random walker algorithm
    segmentation = random_walker(data, markers, beta=100, mode='bf')
    labels = segmentation - 1
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Prostate data')
    ax2.imshow(markers, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Scribble')
    ax3.imshow(labels, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentation')

    fig.tight_layout()
    plt.savefig("../imgs/prostate_randomwalker_segmentation.png")
    plt.show()
    return "End"


if __name__ == "__main__":
    show_example_acdc()
    show_example_protate()
