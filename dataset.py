import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from ultralytics.data.utils import polygon2mask
from skimage.color import rgb2gray

def read_seg(label, imgsz=(512, 512)):
    masks = []
    with open(label, "r") as f:
        for line in f:
            cont = line.split()
            ncls, seg = cont[0], cont[1:]
            seg = np.array(seg).astype(float).reshape((-1, 2))
            masks.append(polygon2mask(
                imgsz,
                [seg * imgsz],
                color=255,
                downsample_ratio=1,
            ))
    return masks


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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images", str(idx + 1) + ".png")
        image = 1 - rgb2gray(io.imread(img_name))
        mask = self.mask[idx + 1]
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
