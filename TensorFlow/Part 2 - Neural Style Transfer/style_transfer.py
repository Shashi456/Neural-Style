
import tensorflow as tf
from keras.applications.vgg19 import VGG19

from keras.models import Model
from keras import backend as K

import numpy as np
import cv2

from utils import load_img, deprocess_img, image_show
from loss import  compute_loss, get_feature_representations, gram_matrix

content_layers = ['block3_conv3']
style_layers   = ['block1_conv1','block2_conv2','block4_conv3']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# path where the content and style images are located
content_path = 'content.jpg'
style_path   = 'style.jpg'

# Save the result as
save_name = 'generated.jpg'

# Vgg weights path
vgg_weights = "vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Using Keras Load VGG19 model
def get_model(content_layers,style_layers):

  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg19           = VGG19(weights=None, include_top=False)
  vgg19.trainable = False

  style_model_outputs   =  [vgg19.get_layer(name).output for name in style_layers]
  content_model_outputs =  [vgg19.get_layer(name).output for name in content_layers]
  
  model_outputs = content_model_outputs+ style_model_outputs

  # Build model 
  return Model(inputs = vgg19.input, outputs = model_outputs),  vgg19


def run_style_transfer(content_path, style_path, num_iterations=200, content_weight=0.1, style_weight=0.9): 

  # Create a tensorflow session 
  sess = tf.Session()
  K.set_session(sess) # Assign keras back-end to the TF session which we created

  model, vgg19 = get_model(content_layers,style_layers)

  # We don't need to (or want to) train any layers of our pre-trained vgg model, so we set it's trainable to false. 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path, num_content_layers)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

  # VGG default normalization
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means 
    

  # In original paper, the initial stylized image is random matrix of same size as that of content image
  # but in later images content image was used instead on random values for first stylized image
  # because it proved to help to stylize faster
  generated_image = load_img(content_path)
  # generated_image = np.random.randint(0,255, size=generated_image.shape) 
  
  # Create tensorflow variable to hold a stylized/generated image during the training 
  generated_image = tf.Variable(generated_image, dtype=tf.float32)

  model_outputs = model(generated_image)

  # weightages of each content and style images i.e alpha & beta
  loss_weights = (style_weight, content_weight)

  # Create our optimizer
  loss = compute_loss(model, loss_weights, model_outputs, gram_style_features, content_features, num_content_layers, num_style_layers)
  opt = tf.train.AdamOptimizer(learning_rate=9, beta1=0.9, epsilon=1e-1).minimize( loss[0], var_list = [generated_image])

  sess.run(tf.global_variables_initializer())
  sess.run(generated_image.initializer)
  
  # loading the weights again because tf.global_variables_initializer() resets the weights
  vgg19.load_weights(vgg_weights)


  # Put loss as infinity before training starts and Create a variable to hold best image (i.e image with minimum loss)
  best_loss, best_img = float('inf'), None

  for i in range(num_iterations):

    # Do optimization
    sess.run(opt)

    # Make sure image values stays in the range of max-min value of VGG norm 
    clipped = tf.clip_by_value(generated_image, min_vals, max_vals)
    # assign the clipped value to the tensor stylized image
    generated_image.assign(clipped)


    # Open the Tuple of tensors 
    total_loss, style_score, content_score = loss
    total_loss = total_loss.eval(session=sess)


    if total_loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = total_loss
      best_img = deprocess_img(generated_image.eval(session=sess))

      # print best loss
      print('\nbest:      iteration: ',i,'   loss: ',total_loss,'  style_loss:    ', style_score.eval(session=sess),'  content_loss:    ',content_score.eval(session=sess),'\n')



    # Save image after every 100 iterations 
    if (i+1)%100 == 0:

      # best_img is in an BGR format, and CV works with images in BGR hence no swapping of channel B and R are required here
      cv2.imwrite(str(i+1)+'-'+save_name,best_img)

  # after num_iterations iterations are completed, close the TF session 
  sess.close()
      
  return best_img, best_loss

best, best_loss = run_style_transfer(content_path, style_path)