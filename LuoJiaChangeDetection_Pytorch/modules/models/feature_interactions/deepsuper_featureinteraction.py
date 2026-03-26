import torch
import torch.nn as nn
import torch.nn.functional as F

class DS_layer(nn.Module):
    def __init__(self, in_d, out_d, stride, output_padding, n_class):
        super(DS_layer, self).__init__()

        self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x



class DeepSuperInteraction(nn.Module):
    def __init__(self, n_class=1):
        super(DeepSuperInteraction, self).__init__()
        self.ds_lyr2 = DS_layer(64, 32, 2, 1, n_class)
        self.ds_lyr3 = DS_layer(64, 32, 4, 3, n_class)

        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        feaslist1, feaslist2 = inputs

        ds2 = self.ds_lyr2(torch.abs(feaslist1[0] - feaslist2[0]))
        ds3 = self.ds_lyr3(torch.abs(feaslist1[1] - feaslist2[1]))

        return self.sigmoid(ds3),self.sigmoid(ds2)
