# pretrained model 
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

def get_model_VGG16(style_layers, content_layers):
  # function that loads the pretrained vgg16 model

  # loading the vgg16 model
  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
  # set it to not trainable
  vgg16.trainable = False
  # get style and content outputs for respective style and content layer(s)
  style_outputs = [vgg16.get_layer(name).output for name in style_layers]
  content_outputs = [vgg16.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  return models.Model(vgg16.input, model_outputs)

def load_and_process_img_vgg16(path_to_img):
  # loads img and applies vgg16 preprocessing, making it suitable for the vgg16 model
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg16.preprocess_input(img)
  return img

def load_img(path_to_img, batch = True):
  # opens the image from its path, resizes it to the for the vgg network required size and returns it as an array
  # per default it also adds a batch dimension (required for inputs of vgg networks)

  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = img_to_array(img)
  
  if batch == True:
    # adding a batch dimension so img can be fed into vgg networks
    img = np.expand_dims(img, axis=0)
  return img