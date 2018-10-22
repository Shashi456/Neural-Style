#Dataset Processing
transform = tf.Compose([
    tf.Resize(512), #Default image_size
    tf.ToTensor(), #Transform it to a torch tensor
    tf.Lambda(lambda x:x[torch.LongTensor([2, 1,0])]), #Converting from RGB to BGR
    tf.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]), #subracting imagenet mean
    tf.Lambda(lambda x: x.mul_(255))
    ])

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

# def load_img(path):
#     img = Image.open(path)
#     img = Variable(transform(img))
#     img = img.unsqueeze(0)
#     return img
def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

# def save_img(img):
#     post = tf.Compose([
#          tf.Lambda(lambda x: x.mul_(1./255)),
#          tf.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
#          tf.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
#          ])
#     img = post(img)
#     img = img.clamp_(0,1)
#     tutils.save_image(img,
#                 '%s/transfer2.png' % ("./images"),
#                 normalize=True)
#     return

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)  

#Gram matrix for neural style different than fast neural style check.
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h*w) #bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)
        # batch1 : bxmxp, batch2 : bxpxn -> bxmxn
        G = torch.bmm(f, f.transpose(1, 2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(h*w)
