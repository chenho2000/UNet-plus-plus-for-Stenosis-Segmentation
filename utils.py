import matplotlib.pyplot as plt
import numpy as np
import torch
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


def accuracy(data_loader, model, device):
    correct = 0
    total = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (preds == y).sum()
            total += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    # print(f"Accuracy: {correct / total}")
    # print(f"Dice score: {dice_score / len(data_loader)}")
    model.train()

    return correct / total, dice_score / len(data_loader)


def get_loaders(train_dir, val_dir, test_dir, batch_size, train_transform=None, val_transform=None, test_transform=None,
                pin_memory=True):
    train_ds = ARCADE(train_dir, transform=train_transform)
    val_ds = ARCADE(val_dir, transform=val_transform)
    test_ds = ARCADE(test_dir, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader
