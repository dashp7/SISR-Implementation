import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory


Glorot = keras.initializers.GlorotUniform(seed=1875)


def ESPCN(sr_factor=3, channels=1):
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, activation= "relu",kernel_initializer=Glorot,padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(64, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(64, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(32, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(32, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(32, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(16, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(16, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(16, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    x = layers.Conv2D(channels * (sr_factor ** 2), 3, activation= "relu",kernel_initializer=Glorot,padding="same")(x)
    outputs = tf.nn.depth_to_space(x, sr_factor)

    return keras.Model(inputs, outputs, name = "ESPCN" )


def EDSR(sr_factor, channels=1, res_blocks=8):
    inputs = keras.Input(shape=(None, None, channels))
    x=inputs
    x = b = layers.Conv2D(64, 3,kernel_initializer=Glorot, padding='same')(x)
    for i in range(res_blocks):
        b = res_edsr(b, 64)
    b = layers.Conv2D(64, 3, padding='same')(b)
    x = layers.Add()([x, b])
    x = layers.Conv2D(channels * (sr_factor ** 2), 3,kernel_initializer=Glorot, padding='same')(x)
    outputs = tf.nn.depth_to_space(x, sr_factor)
    print(x.shape)
    return keras.Model(inputs, outputs, name="EDSR")


def res_edsr(inp, filters):
    x = layers.Conv2D(filters, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(inp)
    x = layers.Conv2D(filters, 3,kernel_initializer=Glorot,padding="same")(x)
    x = layers.Add()([inp, x])
    return x



def SR_RESNET(sr_factor, channels=1, res_blocks=8):
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 3,kernel_initializer=Glorot, padding='same')(inputs)
    x = b = layers.PReLU(shared_axes=[1, 2])(x)
    for i in range(res_blocks):
        b = res_block(b, 64)
    b = layers.Conv2D(64, 3,kernel_initializer=Glorot, padding='same')(b)
    x = layers.Add()([x, b])
    x = layers.Conv2D(channels * (sr_factor ** 2), 3,kernel_initializer=Glorot, padding='same')(x)
    outputs = tf.nn.depth_to_space(x, sr_factor)
    
    return keras.Model(inputs, outputs, name="SR-RESNET")


def res_block(inp, filters):
    x = layers.Conv2D(filters, 3, activation= "relu",kernel_initializer=Glorot,padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(filters, 3,kernel_initializer=Glorot,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([inp, x])
    return x
