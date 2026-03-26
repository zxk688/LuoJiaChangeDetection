import torch
import torch.nn as nn
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    return F.interpolate(input, size, scale_factor, mode, align_corners)

class IdentityHead(nn.Module):
    """Identity Head."""

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]#single_select

        return inputs
    
    def __init__(self, input_transform):
        super(IdentityHead, self).__init__()
        self.input_transform = input_transform
        self.in_index = [0,1,2,3,4]
        self.align_corners = False
    
    def _forward_feature(self, inputs):
        """
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output

