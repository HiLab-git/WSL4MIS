import SimpleITK as sitk
import glob
import os
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import random_walker


def pseudo_label_generator_acdc(data, seed):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta=100, mode='bf')
        pseudo_label = segmentation - 1
    return pseudo_label


def pseudo_label_generator(data, seed):
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
    pseudo_label = random_walker(data, markers, beta=100, mode='bf')
    return pseudo_label-1


for i in sorted(glob.glob("/home/SENSETIME/luoxiangde.vendor/Projects/WSL4MIS/data/ACDC_training/*_scribble.nii.gz"))[2:]:
    print(i.replace("_scribble.nii.gz", ".nii.gz"))
    img_itk = sitk.ReadImage(i.replace("_scribble.nii.gz", ".nii.gz"))
    image = sitk.GetArrayFromImage(img_itk)
    scribble = sitk.GetArrayFromImage(sitk.ReadImage(i))
    pseudo_volumes = np.zeros_like(image)
    for ind, slice_ind in enumerate(range(image.shape[0])):
        if 1 not in np.unique(scribble[ind, ...]) or 2 not in np.unique(scribble[ind, ...]) or 3 not in np.unique(scribble[ind, ...]):
            pass
        else:
            pseudo_volumes[ind, ...] = pseudo_label_generator(
                image[ind, ...], scribble[ind, ...])
    pseudo_volumes_itk = sitk.GetImageFromArray(pseudo_volumes)
    pseudo_volumes_itk.CopyInformation(img_itk)
    sitk.WriteImage(pseudo_volumes_itk, i.replace(
        "_scribble.nii.gz", "_random_walker.nii.gz"))
#
# def show_example_protate():
#     h5f = h5py.File("../imgs/patient004_slice_8.h5", "r")
#     data = h5f["image"][:]
#     seed = h5f["scribble"][:]
#     # in the seed array: 0 means background, 1 to 2 mean class 1 to 2, 4 means: unknown region
#     markers = np.ones_like(seed)
#     markers[seed == 4] = 0
#     markers[seed == 0] = 1
#     markers[seed == 1] = 2
#     markers[seed == 2] = 3
#
#     sigma = 0.35
#     data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                              out_range=(-1, 1))
#
#     # Run random walker algorithm
#     labels = random_walker(data, markers, beta=100, mode='bf')
#     # Plot results
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
#                                         sharex=True, sharey=True)
#     ax1.imshow(data, cmap='gray')
#     ax1.axis('off')
#     ax1.set_title('Prostate data')
#     ax2.imshow(markers, cmap='gray')
#     ax2.axis('off')
#     ax2.set_title('Scribble')
#     ax3.imshow(labels, cmap='gray')
#     ax3.axis('off')
#     ax3.set_title('Segmentation')
#
#     fig.tight_layout()
#     plt.savefig("../imgs/prostate_randomwalker_segmentation.png")
#     plt.show()
#     return "End"
#
#
# if __name__ == "__main__":
#     show_example_acdc()
#     show_example_protate()
