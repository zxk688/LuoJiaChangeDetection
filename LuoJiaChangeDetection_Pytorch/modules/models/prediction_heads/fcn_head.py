import torch
import torch.nn as nn

class FCNHead(nn.Module):

    def __init__(self,   in_channels=64, output_channels=1):
        super(FCNHead, self).__init__()
        self.output_channels = output_channels
        self.conv_final = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
  
    def forward(self, feas):

        out = self.conv_final(feas)
        if self.output_channels > 1:
            output = self.softmax(out)
        else:
            output = self.sigmoid(out)
        return output

        # out = self.conv_final(feas[-1])
        # out = self.sigmoid(out)
        # return out
