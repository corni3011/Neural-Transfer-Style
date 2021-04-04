import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import numpy as np
import pandas as pd
import time
import datetime
import IPython
from IPython.display import clear_output
import os
import cProfile
from tensorflow import keras
from keras import preprocessing
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

from VGG19_Utils import *
from VGG16_Utils import *

def imshow(img, title=None):
  # function to show images that have a batch dimension
  
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

def deprocess_img(processed_img):
  # to obtain a generated img in the style of our content image, we need to reverse the preprocessing steps after the transfer
  x = processed_img.copy()
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
  channels = int(style_layer_output.shape[-1])
  a = tf.reshape(style_layer_output, [-1, channels])
  n = tf.shape(a)[0]
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
  #
  style_loss = 0
  content_loss = 0

  # compute the style loss
  weight_per_style_layer = 1/num_style_layers
  for target_style_gram, gen_style in zip(style_feature_gram, gen_style_features):
    style_loss += weight_per_style_layer * get_style_loss(gen_style[0], target_style_gram)

  # compute the content loss
  weight_per_content_layer = 1/num_content_layers
  for target_content, gen_content in zip(content_feature, gen_content_features):
    content_loss += weight_per_content_layer * get_content_loss(gen_content[0], target_content)

  # the overall loss is then the weighted sum of style and content loss
  overall_loss = style_loss * style_weight + content_loss * content_weight
  
  return overall_loss, style_loss, content_loss

def compute_grads(model, style_weight, content_weight, gen_image, style_feature_gram, content_feature, num_style_layers, num_content_layers):
  # computes the gradients from the overall loss and the current generated image
  with tf.GradientTape() as tape:
    overall_loss, style_loss, content_loss = compute_loss(model, style_weight, content_weight, gen_image, style_feature_gram, content_feature, num_style_layers, num_content_layers)
  return tape.gradient(overall_loss, gen_image), (overall_loss, style_loss, content_loss)

def get_feature_representations(model, content_image, style_image, num_style_layers, num_content_layers):
  style_outputs = model(style_image)
  content_outputs = model(content_image)

  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_content_layers:]]

  return style_features, content_features

def transfer_style(model_type, content_path, style_path, nr_iterations, content_weight, style_weight, content_layers, style_layers):
  # initialise desired vgg model and apply respective preprocessing to content and style image
  if model_type == "vgg19":
    model = get_model_VGG19(style_layers, content_layers)
    content_image = load_and_process_img_vgg19(content_path)
    style_image = load_and_process_img_vgg19(style_path)
  if model_type == "vgg16":
    model = get_model_VGG16(style_layers, content_layers)
    content_image = load_and_process_img_vgg16(content_path)
    style_image = load_and_process_img_vgg16(style_path)
  
  # set model parameters to untrainable
  for layer in model.layers:
    layer.trainable = False


  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  # compute content feature of our content image and the gram matrice of our style image
  style_features, content_features = get_feature_representations(model, content_image, style_image, num_style_layers, num_content_layers)
  style_feature_gram = [get_gram_matrix(style_feature) for style_feature in style_features]

  # generate initial image, i.e. white image
  init_image = tf.Variable(tf.zeros(content_image.shape))
  
  # initialize optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

  # For displaying intermediate images 
  iter_count = 1

  # Store our best result
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
    grads, (loss, style_loss, content_loss) = compute_grads(model, style_weight, content_weight, init_image, style_feature_gram, content_features, num_style_layers, num_content_layers)
    # apply the gradients on our initial image
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals) #???????????????
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
      print("Model currently used: " + model_type)
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


def compare_imgs(vgg19_img, vgg16_img, content_path, style_path):
  # prints the content image and the best generated image by each model used
  plt.figure(figsize=(15,15))

  plt.subplot(2, 2, 1, title = "Original Content", xticks = [], yticks = [])
  plt.imshow(np.squeeze(load_img(content_path).astype('uint8'), axis=0))

  plt.subplot(2, 2, 2, title = "VGG19", xticks = [], yticks = [])
  plt.imshow(vgg19_img)

  plt.subplot(2, 2, 3, title = "VGG16", xticks = [], yticks = [])
  plt.imshow(vgg16_img)

  plt.subplot(2, 2, 4, title = "Original Style", xticks = [], yticks = [])
  plt.imshow(np.squeeze(load_img(style_path).astype('uint8'), axis=0))

  plt.savefig("Comparison.jpg")

def compute_and_visualize_transfer(content_path, style_path, content_layers, style_layers, nr_iterations=1000, content_weight=5, style_weight=0.001):
  
  
  # do transfer once for each model
  best_vgg19_img, best_vgg19_loss = transfer_style("vgg19", content_path, style_path, nr_iterations, content_weight, style_weight, content_layers, style_layers)

  best_vgg16_img, best_vgg16_loss = transfer_style("vgg16", content_path, style_path, nr_iterations, content_weight, style_weight, content_layers, style_layers)

  # compare results of each model
  compare_imgs(best_vgg19_img, best_vgg16_img, content_path, style_path)