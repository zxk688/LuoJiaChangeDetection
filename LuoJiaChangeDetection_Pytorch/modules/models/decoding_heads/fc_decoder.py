import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class FCSiamDecoder(nn.Module):
    def __init__(self, base_channel=16, mode = None):
        super(FCSiamDecoder, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, 
                   base_channel * 8, base_channel * 16]
        self.mode = mode
        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d_diff = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.conv43d_conc = nn.ConvTranspose2d(filters[3]+filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d_diff = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.conv33d_conc = nn.ConvTranspose2d(filters[2]+filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d_diff = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.conv22d_conc = nn.ConvTranspose2d(filters[1]+filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d_EF = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.conv12d_diff = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.conv12d_conc = nn.ConvTranspose2d(filters[0]+filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input1, input2):
        a, b = input1
        c = input2
        x4d = self.upconv4(b[4])
        pad4 = ReplicationPad2d((0, a[3].size(3) - x4d.size(3), 0, a[3].size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), c[3]), 1)

        if self.mode == 'diff':
            x43d = self.do43d(F.relu(self.bn43d(self.conv43d_diff(x4d))))
        elif self.mode == 'concat':
            x43d = self.do43d(F.relu(self.bn43d(self.conv43d_conc(x4d))))

        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, a[2].size(3) - x3d.size(3), 0, a[2].size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), c[2]), 1)

        if self.mode == 'diff':
            x33d = self.do33d(F.relu(self.bn33d(self.conv33d_diff(x3d))))
        elif self.mode == 'concat':
            x33d = self.do33d(F.relu(self.bn33d(self.conv33d_conc(x3d))))

        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, a[1].size(3) - x2d.size(3), 0, a[1].size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), c[1]), 1)

        if self.mode == 'diff':
            x22d = self.do22d(F.relu(self.bn22d(self.conv22d_diff(x2d))))
        elif self.mode == 'concat':
            x22d = self.do22d(F.relu(self.bn22d(self.conv22d_conc(x2d))))

        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, a[0].size(3) - x1d.size(3), 0, a[0].size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), c[0]), 1)

        if self.mode == 'diff':
            x12d = self.do12d(F.relu(self.bn12d(self.conv12d_diff(x1d))))
        elif self.mode == 'concat':
            x12d = self.do12d(F.relu(self.bn12d(self.conv12d_conc(x1d))))

        x11d = self.conv11d(x12d)
        out = self.sigmoid(x11d)
        return out
    
class FCEFDecoder(nn.Module):
    def __init__(self, base_channel=16):
        super(FCEFDecoder, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, 
                   base_channel * 8, base_channel * 16]

        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
       
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(filters[0], 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,a):
        # Stage 4d
        x4d = self.upconv4(a[4])
        pad4 = ReplicationPad2d((0, a[3].size(3) - x4d.size(3), 0, a[3].size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), a[3]), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, a[2].size(3) - x3d.size(3), 0, a[2].size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), a[2]), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, a[1].size(3) - x2d.size(3), 0, a[1].size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), a[1]), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, a[0].size(3) - x1d.size(3), 0, a[0].size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), a[0]), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        out = self.sigmoid(x11d)

        return out
