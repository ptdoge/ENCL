import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change,1
    change,-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, target):

        h, w = distance.shape[-2:]
        if target.shape[-1] != distance.shape[-1] or target.shape[-2] != distance.shape[-2]:
            label = F.interpolate(target, (h, w), mode='nearest')
        else:
            label = target.clone()

        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label == 1).float()) + 0.0001  # 未变化
        neg_num = torch.sum((label == -1).float()) + 0.0001  # 变化

        loss_1 = torch.sum((1 + label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1 - label) / 2 * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)) / neg_num

        loss = loss_1 + loss_2
        return loss