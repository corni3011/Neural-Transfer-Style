import tensorflow as tf
from scipy import signal
import numpy as np
import pandas as pd
import time
import os
from tensorflow import keras
from tensorflow.keras import preprocessing
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow_datasets as tfds
import IPython
import matplotlib.pyplot as plt

from VGG13_Model import *
from VGG13_Style_Transfer import *
from VGG13_Training import *

def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def load_and_process_img_vgg13(path_to_img):
  # loads img and applies vgg19 preprocessing, making it suitable for the vgg19 model
  img = load_img(path_to_img)
  img = VGG13Model.preprocessVGG13(img)
  return img

def deprocess_img(processed_img):
  # function that reverses the vgg preprocessing
  x = processed_img.copy()
  
  # removes possible batch dimension
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")

  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

def get_content_loss(gen_content, target_content):
  # we measure the difference between the pixel values of our generated and the target content representations
  loss = gen_content - target_content
  # we square those differences
  loss = tf.square(loss)
  # and return their mean
  return tf.reduce_mean(loss)

def get_gram_matrix(style_layer_output):
  # computes the gram matrix for one style layer
  channels = int(style_layer_output.shape[-1])
  
  # vectorizes the values for each channel, so it reduces (widht, height, channels) to (width*height, channels)
  a = tf.reshape(style_layer_output, [-1, channels])
  
  # n is width*height and is used to normalize the gram values before returning
  n = tf.shape(a)[0]
  
  # the gram is the matrix multiplication of (width*heigt, channels) and (channels, width*height) so its shape is (channels, channels)
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(gen_style, target_gram):
  # computes the style loss for one style layer
  height, width, channels = gen_style.get_shape().as_list()
  # first we compute the gram matrix of the current style representation for our generated image
  gen_gram = get_gram_matrix(gen_style)
  # then we return the mean squared difference between the gram matrices of our generated and the style image
  return tf.reduce_mean(tf.square(gen_gram - target_gram))# / (4 * (channels ** 2) * (width * height) ** 2)

def compute_loss(model, style_weight, content_weight, gen_image, style_feature_gram, content_feature, num_style_layers, num_content_layers):
  # first we compute the style and content representations of our current generated image
  gen_outputs = model(gen_image)
  
  gen_style_features = gen_outputs[:num_style_layers]
  gen_content_features = gen_outputs[num_content_layers:]
  style_loss = 0
  content_loss = 0
  
  # compute the style loss for all style layers, each style layer is equally weighted
  weight_per_style_layer = 1/num_style_layers
  for target_style_gram, gen_style in zip(style_feature_gram, gen_style_features):
    style_loss += weight_per_style_layer * get_style_loss(gen_style[0], target_style_gram)

  
  # compute the content loss for all content layers (in our case only one), each content layer is equally weighted
  weight_per_content_layer = 1/num_content_layers
  for target_content, gen_content in zip(content_feature, gen_content_features):
    content_loss += weight_per_content_layer * get_content_loss(gen_content[0], target_content)

  # the overall loss is then the weighted sum of style and content loss
  overall_loss = style_loss * style_weight + content_loss * content_weight
  
  return overall_loss, style_loss, content_loss

def compute_grads(model, style_weight, content_weight, gen_image, style_feature_gram, content_feature,  num_style_layers, num_content_layers):
  # computes the gradients from the overall loss and the current generated image
  with tf.GradientTape() as tape:
    overall_loss, style_loss, content_loss = compute_loss(model, style_weight, content_weight, gen_image, style_feature_gram, content_feature,  num_style_layers, num_content_layers)
  return tape.gradient(overall_loss, gen_image), (overall_loss, style_loss, content_loss)

def get_feature_representations(model, content_image, style_image, num_style_layers, num_content_layers):
  # function that computes the content and style features for our content and style image
  style_outputs = model(style_image)
  content_outputs = model(content_image)

  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_content_layers:]]

  return style_features, content_features

def prepare_Model(style_layers, content_layers):
  # initialise the vgg model
  path="""C:\\Users\\Cornelius\\OneDrive\\Downloads\\my_model_weights1617518976.8932934.h5"""
  vgg = VGG13Model((32, 32, 3), 100)
  vgg.import_weights_for_style_transfer(path, style_layers, content_layers)

  return vgg

def transfer_style(content_path, style_path, nr_iterations, content_weight, style_weight, style_layers, content_layers,):
  # initialise desired vgg model and apply respective preprocessing to content and style image

  model = prepare_Model(style_layers, content_layers)

  content_image = load_and_process_img_vgg13(content_path)
  style_image = load_and_process_img_vgg13(style_path)
  
  num_style_layers = len(style_layers)
  num_content_layers = len(content_layers)

  # set model parameters to untrainable
  for layer in model.layers:
    layer.trainable = False

  # compute content feature of our content image and the gram matrice of our style image
  style_features, content_features = get_feature_representations(model, content_image, style_image, num_style_layers, num_content_layers)
  # compute gram matrices for our style image based on its style features
  style_feature_gram = [get_gram_matrix(style_feature) for style_feature in style_features]

  # generate initial image, i.e. black image (all pixel values are zero across all channels)
  init_image = tf.Variable(tf.ones(content_image.shape))
  
  # initialize optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

  # variables for displaying 10 intermediate images
  iter_count = 1

  # Initialize variables to store best result
  best_loss, best_img = float('inf'), None

  # For displaying
  num_rows = 2
  num_cols = 5
  display_interval = nr_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  # initialize list to collect images at displaying intervalls
  imgs = []

  for i in range(nr_iterations):
    iteration_time = time.time()
    # compute the gradients and losses
    grads, (loss, style_loss, content_loss) = compute_grads(model, style_weight, content_weight, init_image, style_feature_gram, content_features,  num_style_layers, num_content_layers)
    # apply the gradients on our initial image
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    # Check if current generated image is the one with the lowest loss and if so replace best_img with current image
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    # check if iteration is at display intervall and if so display current generated image
    if i % display_interval== 0:
      start_time = time.time()
      
      # reverse preprocessing before plotting
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))

      # print information to track progress
      print("Model currently used: VGG13")
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_loss, content_loss, time.time() - start_time, ))
      print("time approx. left with current model: ", round((time.time() - iteration_time)*(1000-i)), "s")
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  IPython.display.clear_output(wait=True)
  
  # we return only the image with the smallest loss and its respective loss
  return best_img, best_loss 

def compare_imgs(vgg19_img, style_path, content_path):
  # prints the content image and the best generated image by each model used
  plt.figure(figsize=(15,15))

  plt.subplot(1, 3, 1, title = "Style", xticks = [], yticks = [])
  plt.imshow(np.squeeze(load_img(style_path).astype('uint8'), axis=0))

  plt.subplot(1, 3, 2, title = "VGG13", xticks = [], yticks = [])
  plt.imshow(vgg19_img)

  plt.subplot(1, 3, 3, title = "Original", xticks = [], yticks = [])
  plt.imshow(np.squeeze(load_img(content_path).astype('uint8'), axis=0))
  plt.savefig("Comparison.jpg")

def compute_and_visualize_transfer(content_path, style_path, style_layers, content_layers, nr_iterations=1000, content_weight=5000, style_weight=0.001):
  
  # do transfer once for each model
  best_vgg19_img, best_vgg19_loss = transfer_style(content_path, style_path, nr_iterations, content_weight, style_weight, style_layers, content_layers,)

  # compare results of each model
  compare_imgs(best_vgg19_img, style_path, content_path)