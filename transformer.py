import torch
import torch.nn as nn

# The architecture for the transformer Net can be found at https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



class ResidualBlock(nn.Module):
# Same Residual Blocks as in Resnet
# Might want to experiment by adding relu/batch norm before returning output
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.relu(self.in1(self.conv1))
        out = self.in2(self.conv2(out))
        out = out + res
        return out


def UpsampleConvLayer(nn.Module):
    #Upsampling layer instead of deconvolution since they give checkerboard effects
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__():
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        #Why Interpolate? This is done instead of using an Upsample Layer
        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class TransformerNet(torch.nn.Module):
    # See supplementary material for architecture
    def __init__(self):
        super(TransformerNet, self).__init__()
        #Convolution Layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.ini1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(3, 64, kernel_size=3, stride=2)
        self.ini2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(3, 128, kernel_size=3, stride=2)
        self.ini3 = nn.InstanceNorm2d(128, affine=True)
        # Residual Blocks, there significance can be read from the original resnet paper
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        #Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.ini4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.ini5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.ini1(self.conv1(x)))
        y = self.relu(self.ini2(self.conv2(y)))
        y = self.relu(self.ini3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.ini4(self.deconv1(y)))
        y = self.relu(self.ini5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
