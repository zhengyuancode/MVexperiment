
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        target_onehot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(input * target_onehot)
        union = torch.sum(input) + torch.sum(target_onehot)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return F.cross_entropy(input, target)