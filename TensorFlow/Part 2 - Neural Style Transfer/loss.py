from utils import load_img
import tensorflow as tf


### Content Loss Function
def get_content_loss(content, target):
  return tf.reduce_mean(tf.square(content - target)) /2



### Style Loss Fucntion
def gram_matrix(input_tensor):

  # if input tensor is a 3D array of size Nh x Nw X Nc
  # we reshape it to a 2D array of Nc x (Nh*Nw)
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]

  # get gram matrix 
  gram = tf.matmul(a, a, transpose_a=True)
  
  return gram

def get_style_loss(base_style, gram_target):

  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  # Original eqn as a constant to divide i.e 1/(4. * (channels ** 2) * (width * height) ** 2)
  return tf.reduce_mean(tf.square(gram_style - gram_target)) / (channels**2 * width * height) #(4.0 * (channels ** 2) * (width * height) ** 2)



### Use to pass content and style image through it 
def get_feature_representations(model, content_path, style_path, num_content_layers):

  # Load our images in 
  content_image = load_img(content_path)
  style_image   = load_img(style_path)
  
  # batch compute content and style features
  content_outputs = model(content_image)
  style_outputs   = model(style_image)
  
  # Get the style and content feature representations from our model  
  style_features   = [ style_layer[0]  for style_layer    in style_outputs[num_content_layers:] ]
  content_features = [ content_layer[0] for content_layer in content_outputs[:num_content_layers] ]

  return style_features, content_features


### Total Loss
def compute_loss(model, loss_weights, generated_output_activations, gram_style_features, content_features, num_content_layers, num_style_layers):

  generated_content_activations = generated_output_activations[:num_content_layers]
  generated_style_activations   = generated_output_activations[num_content_layers:]

  style_weight, content_weight = loss_weights
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, generated_style_activations):
    temp = get_style_loss(comb_style[0], target_style)
    style_score += weight_per_style_layer * temp
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, generated_content_activations):
    temp = get_content_loss(comb_content[0], target_content)
    content_score += weight_per_content_layer* temp

  # Get total loss
  loss = style_weight*style_score + content_weight*content_score 


  return loss, style_score, content_score