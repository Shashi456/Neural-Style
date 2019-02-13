from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datas
import torchvision.transforms as tf
import torchvision.utils as tutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models
from vgg import VGG



#### UTILS ####

#Dataset Processing


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std



#Gram matrix for neural style different than fast neural style check.
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h*w) #bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)
        # batch1 : bxmxp, batch2 : bxpxn -> bxmxn
        G = torch.bmm(f, f.transpose(1, 2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(h*w)


#### THE NETWORK ####

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

    def forward(self, x, out_params = None):
        out = {}
        # Building up the VGG net that's going to be used
        out['re11'] = F.relu(self.conv1_1(x))
        out['re12'] = F.relu(self.conv1_2(out['re11']))
        out['p1'] = self.p1(out['re12'])
        h_relu1_2 = out['re12']
        out['re21'] = F.relu(self.conv2_1(out['p1']))
        out['re22'] = F.relu(self.conv2_2(out['re21']))
        out['p2'] = self.p2(out['re22'])
        h_relu2_2 = out['re22']
        out['re31'] = F.relu(self.conv3_1(out['p2']))
        out['re32'] = F.relu(self.conv3_2(out['re31']))
        out['re33'] = F.relu(self.conv3_3(out['re32']))
        out['re34'] = F.relu(self.conv3_4(out['re33']))
        out['p3'] = self.p3(out['re34'])
        h_relu3_3 = out['re33']
        out['re41'] = F.relu(self.conv4_1(out['p3']))
        out['re42'] = F.relu(self.conv4_2(out['re41']))
        out['re43'] = F.relu(self.conv4_3(out['re42']))
        out['re44'] = F.relu(self.conv4_4(out['re43']))
        h_relu4_3 = out['re43']
        out['p4'] = self.p4(out['re44'])
        out['re51'] = F.relu(self.conv5_1(out['p4']))
        out['re52'] = F.relu(self.conv5_2(out['re51']))
        out['re53'] = F.relu(self.conv5_3(out['re52']))
        out['re54'] = F.relu(self.conv5_4(out['re53']))
        out['p5'] = self.p5(out['re54'])
        if out_params is not None:
             return [out[param] for param in out_params]
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out



#### THE MAIN CODE ####


#We need to define content and style Layers
content_layers = ['re42']
style_layers = ['re11','re21','re31','re41','re51']
#setting seed for pytorch
#torch.manual_seed(random.randint(1, 10000))
#if you want to run the program on cuda then
torch.cuda.manual_seed_all(random.randint(1, 10000))
if not os.path.exists("images/"):
    os.makedirs("images/")
#The below flag allows you to enable the cudnn auto-tuner
#to find the best algorithm for your hardware
cudnn.benchmark = True

#Dataset Processing
transform = tf.Compose([
    tf.Resize(512), #Default image_size
    tf.ToTensor(), #Transform it to a torch tensor
    tf.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    tf.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]), #subracting imagenet mean
    tf.Lambda(lambda x: x.mul_(255))
    ])

def load_img(path):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

def save_img(img):
    post = tf.Compose([
         tf.Lambda(lambda x: x.mul_(1./255)),
         tf.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
         tf.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
         ])
    img = post(img)
    img = img.clamp_(0,1)
    tutils.save_image(img,
                '%s/transfer2.png' % ("./images"),
                normalize=True)
    return

style_img = "./QiBashi.jpg"
content_img = "./2.jpg"
styleImg = load_img(style_img)
contentImg = load_img(content_img)

#for running on cuda
styleImg = styleImg.cuda()
contentImg = contentImg.cuda()

vgg_directory = "./vgg_conv.pth" #path to pretrained vgg vgg_directory
vgg = VGG()
#print(vgg.state_dict())
vgg.load_state_dict(torch.load(vgg_directory))
for param in vgg.parameters():
    param.requires_grad = False

vgg.cuda() # Putting model on cuda

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h*w) #bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)
        # batch1 : bxmxp, batch2 : bxpxn -> bxmxn
        G = torch.bmm(f, f.transpose(1, 2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(h*w)

class styleLoss(nn.Module):
    def forward(self, input, target):
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput, target)

styleTargets = []
for t in vgg(styleImg, style_layers):
    t = t.detach()
    styleTargets.append(GramMatrix()(t))

contentTargets = []
for t in vgg(contentImg, content_layers):
    t = t.detach()
    contentTargets.append(t)

style_Losses = [styleLoss()] * len(style_layers)

content_Losses = [nn.MSELoss()] * len(content_layers)

# We only need to go through the vgg once to get all style and content losses

losses = style_Losses + content_Losses
targets = styleTargets + contentTargets
loss_layers = style_layers + content_layers
style_weight = 1000
content_weight = 5
weights = [style_weight] * len(style_layers) + [content_weight] * len(content_layers)

optimImg = Variable(contentImg.data.clone(), requires_grad=True)
optimizer = optim.LBFGS([optimImg])

#Shifting everything to cuda
for loss in losses:
    loss = loss.cuda()
optimImg.cuda()

# Training
no_iter = 100

for iteration in range(1, no_iter):
    print('Iteration [%d]/[%d]'%(iteration,no_iter))
    def cl():
        optimizer.zero_grad()
        out = vgg(optimImg, loss_layers)
        totalLossList = []
        for i in range(len(out)):
            layer_output = out[i]
            loss_i = losses[i]
            target_i = targets[i]
            totalLossList.append(loss_i(layer_output, target_i) * weights[i])
        totalLoss = sum(totalLossList)
        totalLoss.backward()
        print('Loss: %f'%(totalLoss.data[0]))
        return totalLoss
    optimizer.step(cl)
outImg = optimImg.data[0].cpu()
save_img(outImg.squeeze())
