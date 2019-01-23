# Neural Style Transfer

## Neural Style Transfer Implementation - Gatys et al 
This section implements the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), this section includes code heavily borrowed from Leon Gatys' [implementation](https://github.com/leongatys/PytorchNeuralStyleTransfer) of the same.

Medium article [link](https://medium.com/@pawsed)

## PyTorch

### Prerequisites:
- Pytorch
- Python

To run: change ```style_img```, ```content_img``` and ```vgg_directory``` locations. The vgg directory needs to include the weights which can be downloaded from [here](https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth). The result will be stored in the ```images``` directory with the name ```transfer.png```. The ```style_weight```, ```content_weight``` and ```no_iter```(No of iterations) can be changed according to user preference

```
python train.py
```
> The following images were run with 0.5 weight for content and style and 100 iterations each.

<img src="https://user-images.githubusercontent.com/18056781/45930315-3de3f680-bf7c-11e8-84df-8d52938fb42c.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930317-3fadba00-bf7c-11e8-8b0a-8b8d956cd041.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930321-49372200-bf7c-11e8-9030-c31e9c9b8636.png" width="256">

<img src="https://user-images.githubusercontent.com/18056781/45930320-463c3180-bf7c-11e8-916f-fd170540e37c.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930319-43d9d780-bf7c-11e8-9548-3b1a49abdb05.jpg" width="256" height="350"> <img src="https://user-images.githubusercontent.com/18056781/45930322-4b00e580-bf7c-11e8-90aa-4d3595fb0e40.png" width="256">

-----------------------------------------

## TensorFlow

### Prerequisites
- TensorFlow
- Python
- Keras ( Only used to load VGG19 model )

Example to show case how easy it is to load VGG19 using Keras
```python 
from keras.applications.vgg19 import VGG19

weights     = "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
vgg19_model = VGG19(weights=weights, include_top=False)
```

### Training process

To run : python style_transfer.py

```style_path``` & ```content_path```  Change them according to your content and style image paths. The vgg directory (i.e ```vgg_weights``` ) needs to include the weights which can be downloaded from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5).  The result will be stored has indicated by ```save_name``` variable.

```python 

# path where the content and style images are located
content_path = 'content.jpg'
style_path   = 'style.jpg'

# Save the result as
save_name = 'generated.jpg'

# Vgg weights path
vgg_weights = "vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

```

By default number of iterations is set to 200, While content weight is set to 0.1 and style weight is set to 0.9.
```python

# functions default definition
run_style_transfer(content_path, style_path, num_iterations=200, content_weight=0.1, style_weight=0.9)

```

### Following examples were trained on default settings

![example 1](https://user-images.githubusercontent.com/26245515/49684223-d0fde880-faf6-11e8-9137-f311b9433bf6.jpg)


![example 2](https://user-images.githubusercontent.com/26245515/49684224-d4916f80-faf6-11e8-8ec0-d9d2ad5c0a06.jpg)

