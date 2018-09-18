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

])
