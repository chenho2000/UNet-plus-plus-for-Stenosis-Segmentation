import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from dataset import ARCADE


def plot_image_with_mask(image, mask):
    """
    Plots an image with its corresponding mask.
    """
    plt.imshow(image, cmap='gray')
    masked_image = np.ma.masked_equal(mask, 0)
    plt.imshow(masked_image, cmap='gray')
    plt.show()


def calculate_f1_score(output, target):
    tp = np.sum((target == 1) & (output == 1))
    fp = np.sum((target == 0) & (output == 1))
    fn = np.sum((target == 1) & (output == 0))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    f1_score = 2 * (precision * recall + 1e-7) / (precision + recall + 1e-7)

    return f1_score


def iou_score(output, target):
    intersection = (output & target).sum()
    union = (output | target).sum()
    return intersection / (union + 1e-7)


def dice(output, target):
    output = output.flatten()
    target = target.flatten()
    intersection = (output * target).sum()

    return (2. * intersection + 1e-7) / (output.sum() + target.sum() + 1e-7)


def track_metric(data_loader, model, device):
    total_f1 = 0
    total_iou = 0
    total_dsc = 0
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.numpy().reshape((512, 512)).astype(int)
            pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy().reshape((512, 512)).astype(int)
            f1 = f1_score(pred, y, average='micro')
            iou = jaccard_score(pred, y, average='micro')
            dsc = dice(pred, y)
            total_f1 += f1
            total_iou += iou
            total_dsc += dsc
    avg_f1 = total_f1 / len(data_loader)
    avg_iou = total_iou / len(data_loader)
    avg_dsc = total_dsc / len(data_loader)
    model.train()

    return avg_f1, avg_iou, avg_dsc


def get_loaders(train_dir, val_dir, test_dir, batch_size, train_transform=None, val_transform=None, test_transform=None,
                pin_memory=True):
    train_ds = ARCADE(train_dir, transform=train_transform, train=True)
    val_ds = ARCADE(val_dir, transform=val_transform)
    test_ds = ARCADE(test_dir, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader
