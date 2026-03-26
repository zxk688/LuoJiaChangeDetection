
import torch
import torch.nn as nn
import torch.nn.functional as F

'''BCEWLLoss'''
class BCEWLLoss(nn.Module):
    def __init__(self, scale_factor=1.0, temperature=1, reduction='mean', log_target=False, lowest_loss_value=None):
        super(BCEWLLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss