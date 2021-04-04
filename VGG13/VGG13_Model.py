import tensorflow as tf
from scipy import signal
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import preprocessing
import image
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

import keras.engine.training as training

from tensorflow.keras import Model
from keras.layers import Dense, AveragePooling2D, Input, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class VGG13Model(tf.keras.Model):

  # the constructer creates the fundamental network structure
  def __init__(self, input_shape, nr_classes):
    super(VGG13Model, self).__init__()

    input = (Input(shape=input_shape))
    model = (tf.keras.layers.experimental.preprocessing.Resizing(256, 256))(input)
    model = self.vggBlock(model, 64, 1, 1)
    model = self.vggBlock(model, 128, 1, 2)
    model = self.vggBlock(model, 256, 2, 3)
    model = self.vggBlock(model, 512, 2, 4)
    model = self.vggBlock(model, 512, 2, 5)
    model = (AveragePooling2D((2,2), strides=(2,2)))(model)
    model = (Flatten())(model)
    model = (Dense(units=4096, activation='relu'))(model)
    model = (Dense(units=4096, activation='relu'))(model)
    model = (Dense(units=nr_classes, activation='softmax'))(model)

    opt = Adam(lr=0.001)
    self.model = training.Model(input, model, name='vgg13')

  # calls the model and returns an output
  def call(self, input):
    result = self.model(input)
    return result

  # compiles the network which is created by the constructor
  def compile(self):
    opt = Adam(lr=0.001)
    self.model.compile(opt, 'categorical_crossentropy', metrics=["accuracy"])

  # Adds one Block to the VGG Network
  def vggBlock(self, model, n_filters, n_conv, block_nr):
    for i in range(n_conv):
      model = (Conv2D(n_filters, (3,3), padding='same', activation='relu', name='block'+str(block_nr)+'_conv'+str(i+1)))(model)
    model = (AveragePooling2D((2,2), strides=(2,2)))(model)
    return model


  # takes a dataset and trains on it
  def train(self, dataset, n_episodes, batchsize):
    callback = tf.keras.callbacks.LearningRateScheduler(VGG13Model.__scheduler)
    self.model.fit(dataset, batch_size=batchsize, epochs=n_episodes, callbacks=[callback])

  # imports weights and sets the expected style and content output layers. Compiles afterwards
  def import_weights_for_style_transfer(self, weights, style_layers, content_layers):
    style_outputs = [self.model.get_layer(name).output for name in style_layers]
    content_outputs = [self.model.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    opt = Adam(lr=0.001)

    self.model.load_weights(weights)
    self.model = training.Model(self.model.input, model_outputs, name='vgg13')
    self.model.compile(opt, 'categorical_crossentropy')


  # returns weights of the model
  def export_model(self, path):
    self.model.save_weights(path)

  # Rezises the input image to the expected size
  @staticmethod
  def preprocessVGG13(img):
    img = tf.image.resize(img,size=[32, 32])
    return img.numpy()
    
  # schedular which changes the learning rate every 15 epochs. For internal use of the model only
  @staticmethod
  def __scheduler(epoch, lr):
    if epoch % 15 == 0 and epoch != 0:
      lr = lr * 0.1
      print(lr)
      return lr
    else:
      return lr