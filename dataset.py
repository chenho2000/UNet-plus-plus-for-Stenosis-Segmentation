import os

import numpy as np
import torch
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset
from ultralytics.data.utils import polygon2mask


def read_seg(label, imgsz=(512, 512)):
    masks = np.zeros(imgsz, dtype=float)
    with open(label, "r") as f:
        for line in f:
            cont = line.split()
            ncls, seg = cont[0], cont[1:]
            seg = np.array(seg).astype(float).reshape((-1, 2))

            curr = polygon2mask(
                imgsz,
                [seg * imgsz],
                color=1,
                downsample_ratio=1,
            )

            # combine masks
            masks = 1 * ((masks == 1) | (curr == 1))

    return masks.astype(float)


class ARCADE(Dataset):
    """ARCADE dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images/labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask = dict()
        self.root_dir = root_dir
        self.transform = transform
        label_path = os.path.join(root_dir, "labels")
        for i in os.listdir(os.path.join(root_dir, "labels")):
            if i.endswith(".txt"):
                self.mask[int(i[:-4])] = read_seg(os.path.join(label_path, i))

    def __len__(self):
        return len(self.mask.keys())

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, "images", str(idx + 1) + ".png")

        image = io.imread(img_name)

        # Convert to grayscale if the image is not already
        if image.ndim == 2:
            image = image / 255
        else:
            image = rgb2gray(io.imread(img_name))

        if idx + 1 in self.mask.keys():
            mask = self.mask[idx + 1]
        else:
            mask = np.zeros((512, 512))

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image.reshape(1, 512, 512), mask.reshape(1, 512, 512)
