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

# pretrained model 
def get_model_VGG19(style_layers, content_layers):
  # function that loads the pretrained vgg19 model

  # loading the vgg19 model
  vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg19.trainable = False
  # get style and content outputs for respective style and content layer(s)
  style_outputs = [vgg19.get_layer(name).output for name in style_layers]
  content_outputs = [vgg19.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  return models.Model(vgg19.input, model_outputs)

def load_and_process_img_vgg19(path_to_img):
  # loads img and applies vgg19 preprocessing, making it suitable for the vgg19 model
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def load_img(path_to_img, batch = True):
  # opens the image from its path, resizes it to the for the vgg network required size and returns it as an array
  # per default it also adds a batch dimension (required for inputs of vgg networks)
  max_dim = 512
  img = Image.open(path_to_img)
  #print(img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = img_to_array(img)
  
  if batch == True:
    # adding a batch dimension so img can be fed into vgg networks
    img = np.expand_dims(img, axis=0)
  return img