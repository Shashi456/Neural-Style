import torch.nn as nn
import torch.nn.functional as f

#Writing the VGG network
class VGG(nn.Module):
    def __init__(self): #Can have an optional pooling parameter to make it average or max
        super(VGG,self).__init__()
        ##VGG layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        #Pooling Layers : The orignal paper mentioned average Pooling
        self.p1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.p2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.p3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.p4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.p5 = nn.AvgPool2d(kernel_size=2, stride=2)
