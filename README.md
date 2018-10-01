# Neural-Style

### Prerequisites:
- Pytorch
- Python

## Neural Style Transfer Implementation - Gatys et al 
This section implements the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), this section includes code heavily borrowed from Leon Gatys' [implementation](https://github.com/leongatys/PytorchNeuralStyleTransfer) of the same.

To run: change ```style_img```, ```content_img``` and ```vgg_directory``` locations. The vgg directory needs to include the weights which can be downloaded from [here](https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth). The result will be stored in the ```images``` directory with the name ```transfer.png```. The ```style_weight```, ```content_weight``` and ```no_iter```(No of iterations) can be changed according to user preference

```
python train.py
```
> The following images were run with 0.5 weight for content and style and 100 iterations each.

<img src="https://user-images.githubusercontent.com/18056781/45930315-3de3f680-bf7c-11e8-84df-8d52938fb42c.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930317-3fadba00-bf7c-11e8-8b0a-8b8d956cd041.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930321-49372200-bf7c-11e8-9030-c31e9c9b8636.png" width="256">

<img src="https://user-images.githubusercontent.com/18056781/45930320-463c3180-bf7c-11e8-916f-fd170540e37c.jpg" width="256"> <img src="https://user-images.githubusercontent.com/18056781/45930319-43d9d780-bf7c-11e8-9548-3b1a49abdb05.jpg" width="256" height="350"> <img src="https://user-images.githubusercontent.com/18056781/45930322-4b00e580-bf7c-11e8-90aa-4d3595fb0e40.png" width="256">
