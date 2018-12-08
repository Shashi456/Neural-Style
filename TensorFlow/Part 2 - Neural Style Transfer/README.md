# Neural Style Transfer - TensorFlow

### Prerequisites
- TensorFlow
- Python

Orginal Paper : [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

Medium article [link](https://medium.com/@pawsed)

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
run_style_transfer(content_path, style_path, num_iterations=500, content_weight=0.1, style_weight=0.9)

```

### Following examples were trained on default settings
![example 1](https://user-images.githubusercontent.com/26245515/49684223-d0fde880-faf6-11e8-9137-f311b9433bf6.jpg)
![example 2](https://user-images.githubusercontent.com/26245515/49684224-d4916f80-faf6-11e8-8ec0-d9d2ad5c0a06.jpg)

