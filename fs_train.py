from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
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

#We need to define content and style Layers
content_layers = ['re42']
style_layers = ['re11','re22','re31','re43'] #can be experimented with
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
    tf.CenterCrop(512),
    #tf.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    #tf.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]), #subracting imagenet mean
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
