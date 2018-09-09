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

    def forward(self, x, out_params):
        out = {}
        # Building up the VGG net that's going to be used
        out['re11'] = f.relu(self.conv1_1(x))
        out['re12'] = F.relu(self.conv1_2(out['re11']))
        out['p1'] = self.pool1(out['re12'])
        out['re21'] = F.relu(self.conv2_1(out['p1']))
        out['re22'] = F.relu(self.conv2_2(out['re21']))
        out['p2'] = self.pool2(out['re22'])
        out['re31'] = F.relu(self.conv3_1(out['p2']))
        out['re32'] = F.relu(self.conv3_2(out['re31']))
        out['re33'] = F.relu(self.conv3_3(out['re32']))
        out['re34'] = F.relu(self.conv3_4(out['re33']))
        out['p3'] = self.pool3(out['re34'])
        out['re41'] = F.relu(self.conv4_1(out['p3']))
        out['re42'] = F.relu(self.conv4_2(out['re41']))
        out['re43'] = F.relu(self.conv4_3(out['re42']))
        out['re44'] = F.relu(self.conv4_4(out['re43']))
        out['p4'] = self.pool4(out['re44'])
        out['re51'] = F.relu(self.conv5_1(out['p4']))
        out['re52'] = F.relu(self.conv5_2(out['re51']))
        out['re53'] = F.relu(self.conv5_3(out['re52']))
        out['re54'] = F.relu(self.conv5_4(out['re53']))
        out['p5'] = self.pool5(out['re54'])
        return [out[param] for param in out_params]
