from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

def build_model(im_shape, vocab_size, num_answers, big_model):
  # The CNN
  im_input = Input(shape=im_shape)
#   x1 = Conv2D(8, 3, padding='same')(im_input)
#   x1 = MaxPooling2D()(x1)
#   x1 = Conv2D(16, 3, padding='same')(x1)
#   x1 = MaxPooling2D()(x1)
#   if big_model:
#     x1 = Conv2D(32, 3, padding='same')(x1)
#     x1 = MaxPooling2D()(x1)
#   x1 = Flatten()(x1)
#   #Load model wothout classifier/fully connected layers
  x1 = VGG16(weights='imagenet', include_top=False, input_shape=(250, 250, 3))(im_input)
  #width_shape = 256
  #height_shape = 256

  #image_input = Input(shape=(width_shape, height_shape, 3))
  #x1 = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
  #print("model type=",type(x1)," Size=",x1.shape)
  #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#   for layer in x1.layers:
# 	  layer.trainable = False

  x1 = tf.keras.layers.Activation('relu', name='relu_conv1')(x1)
  x1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x1)
  x1 = tf.keras.layers.Convolution2D(3, 1, 1, name='conv2')(x1)
  x1 = tf.keras.layers.Activation('relu', name='relu_conv2')(x1)
  x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
  x1= Dense(32, activation= 'tanh')(x1)

  print("shape of x1",x1.shape)
  # The question network
  q_input = Input(shape=(vocab_size,))
  x2 = Dense(32, activation='tanh')(q_input)
  x2 = Dense(32, activation='tanh')(x2)
  print("shape of x2",x2.shape)
  print("type of x1 and x2", type(x1), type(x2))

  # Merge -> output
  out = Multiply()([x1, x2])
  out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

  return model
