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
from torch.utils.data import Dataloader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models
from vgg import VGG
from transformer import TransformerNet
from utils import normalize_batch, load_image, save_image

#We need to define content and style Layers
content_layers = ['re42']
style_layers = ['re11','re22','re31','re43'] #can be experimented with
#setting seed for pytorch
#torch.manual_seed(random.randint(1, 10000))
#if you want to run the program on cuda then
save_model_dir = "./models/"
torch.cuda.manual_seed_all(random.randint(1, 10000))
if not os.path.exists("images/"):
    os.makedirs("images/")
if not os.path.exists("models/"):
    os.makedirs("models/")
#The below flag allows you to enable the cudnn auto-tuner
#to find the best algorithm for your hardware
cudnn.benchmark = True
device = torch.device("cuda")

#Dataset Processing
transform = tf.Compose([
    tf.Resize(256), #Default image_size
    tf.ToTensor(), #Transform it to a torch tensor
    tf.CenterCrop(256),
    #tf.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    #tf.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[0.225, 0.224, 0.229]), #subracting imagenet mean
    tf.Lambda(lambda x: x.mul_(255))
    ])

data = "./data/"
train_dataset = datas.ImageFolder(data, transform)
train_loader = Dataloader(train_dataset, batch_size=4)

transformer = TransformerNet().to(device)
optimizer = Adam(transformer.parameters(), 1e-3)
mse_loss = nn.MSELoss()

vgg_directory = "./vgg_conv.pth" #path to pretrained vgg vgg_directory
vgg = VGG()
vgg.load_state_dict(torch.load(vgg_directory))
for param in vgg.parameters():
    param.requires_grad = False

vgg.to(device) # Putting model on cuda

def load_img(path):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

style_img = "./QiBashi.jpg"

styleImg = load_img(style_img)
styleImg = styleImg.repeat(4, 1, 1, 1).to(device) # Its actually repeat(<batch_size>,1,1,1)

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


#contentImg = load_img(content_img)

#for running on cuda
#styleImg = styleImg.cuda()
#contentImg = contentImg.cuda()





# class GramMatrix(nn.Module):
#     def forward(self, input):
#         b, c, h, w = input.size()
#         f = input.view(b, c, h*w) #bxcx(hxw)
#         # torch.bmm(batch1, batch2, out=None)
#         # batch1 : bxmxp, batch2 : bxpxn -> bxmxn
#         G = torch.bmm(f, f.transpose(1, 2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
#         return G.div_(h*w)
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class styleLoss(nn.Module):
    def forward(self, input, target):
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput, target)

features_style = vgg(normalize_batch(styleImg), style_layers)
gram_style = [gram_matrix(y) for y in features_style]

styleTargets = []
#for t in vgg(styleImg, style_layers):
#    t = t.detach()
#    styleTargets.append(GramMatrix()(t))

#contentTargets = []
#for t in vgg(contentImg, content_layers):
#    t = t.detach()
#    contentTargets.append(t)

#style_Losses = [styleLoss()] * len(style_layers)

#content_Losses = [nn.MSELoss()] * len(content_layers)

# We only need to go through the vgg once to get all style and content losses

#losses = style_Losses + content_Losses
#targets = styleTargets + contentTargets
#loss_layers = style_layers + content_layers
#style_weight = 1000
# content_weight = 5
# weights = [style_weight] * len(style_layers) + [content_weight] * len(content_layers)
#
# #optimImg = Variable(contentImg.data.clone(), requires_grad=True)
# #optimizer = optim.LBFGS([optimImg])
#
# #Shifting everything to cuda
# for loss in losses:
#     loss = loss.cuda()
# optimImg.cuda()
#
# Training
no_iter = 2 #default 2
content_weight = 1e5
style_weight = 1e10

for iteration in range(1, no_iter):
    print('Iteration [%d]/[%d]'%(iteration,no_iter))
    # def cl():
    #     optimizer.zero_grad()
    #     out = vgg(optimImg, loss_layers)
    #     totalLossList = []
    #     for i in range(len(out)):
    #         layer_output = out[i]
    #         loss_i = losses[i]
    #         target_i = targets[i]
    #         totalLossList.append(loss_i(layer_output, target_i) * weights[i])
    #     totalLoss = sum(totalLossList)
    #     totalLoss.backward()
    #     print('Loss: %f'%(totalLoss.data[0]))
    #     return totalLoss
    # optimizer.step(cl)
    transformer.train()
    total_content_loss = 0.
    total_style_loss = 0.
    count = 0
    for batch_id, (x, _) in enumerate(train_loader):
        b_len = len(x) #batch_len
        count += b_len
        optimizer.zero_grad()

        x = x.to(device)
        y = transformer(y)

        x = normalize_batch(x)
        y = normalize_batch(y)

        features_y = vgg(y)
        features_x = vgg(x)

        content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0

        for feature_y, gram_s in zip(features_y, gram_style):
            gram_y = gram_matrix(feature_y)
            style_loss = mse_loss(gram_y, gram_s[:b_len, :, :])

        style_lose *= style_weight

        total_loss = content_loss + style_loss

        total_loss.backward()
        optimizer.step()

        total_content_loss += content_loss.item()
        total_style_loss += style_loss.item()

        if (batch_id + 1) % 1000 == 0:
            print("{}\tEpoch {}:\t[{}/{}]\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  total_loss / (batch_id + 1)))

#Save the Model

transformer.eval().cpu()
save_model_filename = str(style_img.strip('.img')) + "_style" + ".pth"
save_model_path = os.path.join(save_model_dir, save_model_filename)

torch.save(transformer.state_dict(), save_model_path)



## Stylization
#content_img = "./2.jpg"

model = save_model_path
content_image = load_image(args.content_image, scale=256)
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to(device


with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()

save_image("style", output[0])
