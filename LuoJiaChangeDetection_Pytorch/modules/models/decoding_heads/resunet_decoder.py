import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnetDecoder(nn.Module):
    def __init__(self, filters=[32,64, 128, 256, 512]):
        super(ResUnetDecoder, self).__init__()
        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.up_residual_conv5 = ResidualConv(filters[0],filters[0],1,1)
        
    def forward(self, inputs):
        x5,fis = inputs
        # Decode
        x5 = self.upsample_1(x5) #512*20*20
        
        x6 = torch.cat([x5,fis[0]],dim=1) # 768*20*20
        x7 = self.up_residual_conv1(x6) #256*20*20

        x7 = self.upsample_2(x7) #256*40*40

        x8 = torch.cat([x7, fis[1]], dim=1) #384*40*40

        x9 = self.up_residual_conv2(x8) #128*40*40

        x9 = self.upsample_3(x9) #128*80*80
        x10 = torch.cat([x9, fis[2]], dim=1) #192*80*80

        x11 = self.up_residual_conv3(x10) #64*80*80

        x11 = self.upsample_4(x11) #64*160*160
        x12 = torch.cat([x11, fis[3]], dim=1) #96*160*160

        x13 = self.up_residual_conv4(x12) 
        x14 = self.up_residual_conv5(x13)
        # output = self.output_layer(x14)
        return (x14,)
