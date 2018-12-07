import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image
import numpy as np
import cv2

def load_img(path_to_img):
  max_dim  = 512
  img      = Image.open(path_to_img)
  img_size = max(img.size)
  scale    = max_dim/img_size
  img      = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

  
  img      = kp_image.img_to_array(img)

  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  
  img = tf.keras.applications.vgg19.preprocess_input(img)

  return tf.convert_to_tensor(img)

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68



  x = np.clip(x, 0, 255).astype('uint8')
  return x

def image_show(img):
  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()