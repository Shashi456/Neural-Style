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
style_layers = ['re11','re21','re31','re41','re51']
#setting seed for pytorch
#torch.manual_seed(random.randint(1, 10000))
#if you want to run the program on cuda then
torch.cuda.manual_seed_all(random.randint(1, 10000))
os.makedirs("images/")
#The below flag allows you to enable the cudnn auto-tuner
#to find the best algorithm for your hardware
cudnn.benchmark = True

#Dataset Processing
transform = tf.Compose([
    tf.Resize(512), #Default image_size
    tf.ToTensor(),
    tf.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    tf.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]), #subracting imagenet mean
    tf.Lambda(lambda x: x.mul_(255))
    ])

def load_img(path):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

def save_img(img):
    post = tf.Compose([
         tf.Lambda(la
                   mbda x: x.mul_(1./255)),
         tf.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
         tf.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
         ])
    img = post(img)
    img = img.clamp_(0,1)
    tutils.save_image(img,
                '%s/transfer.png' % (opt.outf),
                normalize=True)
    return

styleImg = load_img(style_img)
contentImg = load_img(content_img)

#for running on cuda
styleImg = styleImg.cuda()
contentImg = contentImg.cuda()

vgg_directory = "" #path to pretrained vgg vgg_directory
vgg = VGG()
vgg.load_state_dict(torch.load(vgg_directory))
for param in vgg.parameters():
    param.requires_grad = False

vgg.cuda() # Putting model on cuda

def GramMatrix(nn.Module):
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

style_Losses = []
