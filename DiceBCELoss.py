import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Dice + BCE loss with smoothness term.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-7):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
