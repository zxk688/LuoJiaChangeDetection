from luojianet import nn
import luojianet.ops as F

class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch) - 1

        self.model = nn.SequentialCell(
            nn.Conv2d(ch[0], ch[0 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[0+1]),
            nn.ReLU(),
            nn.Conv2d(ch[1], ch[1 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[1+1]),
            nn.ReLU(),
        )
        # for i in range(self.conv_number):
        #     self.model.insert_child_to_cell('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 11, 1, 'pad', 5))
        #     self.model.insert_child_to_cell('bn{0}'.format(i),nn.BatchNorm2d(ch[i+1]))
        #     self.model.insert_child_to_cell('relu{0}'.format(i),nn.ReLU())

    def forward(self, x):
        y = self.model(x)
        return y

class funnel(nn.Module):
    def __init__(self, *ch):
        super(funnel, self).__init__()
        self.conv_number = len(ch) - 1
  
        self.model = nn.SequentialCell(
            nn.Conv2d(ch[0], ch[0 + 1], 5, 1, 'pad', 2),
            nn.BatchNorm2d(ch[0+1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(ch[1], ch[1 + 1], 5, 1, 'pad', 2),
            nn.BatchNorm2d(ch[1+1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(ch[2], ch[2 + 1], 5, 1, 'pad', 2),
            nn.BatchNorm2d(ch[2+1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

        # for i in range(self.conv_number):
        #     self.model.insert_child_to_cell('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 5, 1, 'pad', 2))
        #     self.model.insert_child_to_cell('bn{0}'.format(i),nn.BatchNorm2d(ch[i+1]))
        #     self.model.insert_child_to_cell('relu{0}'.format(i),nn.ReLU())
        #     self.model.insert_child_to_cell('pooling{0}'.format(i),nn.AvgPool2d(kernel_size=2,stride=2))

    def forward(self, x):
        y = self.model(x)
        return y
    
class MultiresolutionEncoder(nn.Module):
    def __init__(self):
        super(MultiresolutionEncoder, self).__init__()
        self.conv0 = funnel(*[3, 4, 6, 9])
        self.conv1 = convx2(*[12, 16, 16]) # sat+edge+cat_from funnel
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = convx2(*[16, 32, 32])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = convx2(*[32, 64, 64, 64])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = convx2(*[64, 128, 128, 128])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, inputs):
        img1 = inputs[:,0]
        img2 = inputs[:,1]
        img_0=self.conv0(img2)
        h1 = self.conv1(F.cat((img1,img_0), 1))
        
        h = self.pool1(h1)
        h2 = self.conv2(h)
        h = self.pool2(h2)
        h3 = self.conv3(h)
  
        h = self.pool3(h3)
        h4 = self.conv4(h)
        h = self.pool4(h4)

        return (h1,h2,h3,h4,h,)
    

    