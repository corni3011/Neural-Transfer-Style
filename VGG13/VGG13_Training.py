import tensorflow as tf
from scipy import signal
import numpy as np
import pandas as pd
import time
import os
from tensorflow import keras
from tensorflow.keras import preprocessing
import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from VGG13_Model import VGG13Model
from VGG13_Utils import *

import tensorflow_datasets as tfds

def preprocess_dataset(train_Images, train_Label):
    train_Label = tf.keras.utils.to_categorical(train_Label)

    images = tf.data.Dataset.from_tensor_slices(train_Images)
    label = tf.data.Dataset.from_tensor_slices(train_Label)

    dataset_final = tf.data.Dataset.zip((images, label))

    dataset_final = dataset_final.shuffle(buffer_size=128)
    dataset_final = dataset_final.batch(128)
    dataset_final = dataset_final.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_final

if __name__ == '__main__':
    model = VGG13Model((32, 32, 3), 100)
    (train_Images, train_Label), (validate_Images, validate_Label) = tf.keras.datasets.cifar100.load_data()
    dataset = preprocess_dataset(train_Images, train_Label)
    model.compile()
    model.train(dataset, 1, 128)
    model.export_model("Weights.ckpt")