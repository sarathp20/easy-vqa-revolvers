from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

def build_model(im_shape, vocab_size, num_answers, big_model):
  # The CNN
  im_input = Input(shape=im_shape)
#   x1 = Conv2D(8, 3, padding='same')(im_input) 8-no of filters, 3-kernel/filter-size
#   x1 = MaxPooling2D()(x1)
#   x1 = Conv2D(16, 3, padding='same')(x1)
#   x1 = MaxPooling2D()(x1)
#   if big_model:
#     x1 = Conv2D(32, 3, padding='same')(x1)
#     x1 = MaxPooling2D()(x1)
#   x1 = Flatten()(x1)
#   #Load model wothout classifier/fully connected layers
  x1 = VGG16(weights='imagenet', include_top=False, input_shape=(250, 250, 3))(im_input)
  x1 = Dropout(0.6)(x1)
  x1 = BatchNormalization(axis=1)(x1)

  #image_input = Input(shape=(250, 250, 3))
  #x1 = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
  #print("model type=",type(x1)," Size=",x1.shape)
  #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#   for layer in x1.layers:
#     layer.trainable = False

  x1 = tf.keras.layers.Activation('relu', name='relu_conv1')(x1)
  x1 = Dropout(0.6)(x1)
  x1 = BatchNormalization(axis=1)(x1)
  x1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x1)
  x1 = BatchNormalization(axis=1)(x1)
  x1 = tf.keras.layers.Convolution2D(3, 1, 1, name='conv2')(x1)
  x1 = Dropout(0.6)(x1)
  x1 = BatchNormalization(axis=1)(x1)
  x1 = tf.keras.layers.Activation('relu', name='relu_conv2')(x1)
  x1 = Dropout(0.6)(x1)
  x1 = BatchNormalization(axis=1)(x1)
  x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
  x1= Dense(512, activation= 'tanh')(x1)
  print("shape of x1",x1.shape)
  # The question network
  q_input = Input(shape=(768,))
  x2 = Dense(512, activation='tanh')(q_input)
  x2 =Dropout(0.6)(x2)
  x2 = BatchNormalization(axis=1)(x2)
  x2 = Dense(512, activation='tanh')(x2)
  print("shape of x2",x2.shape)
  print("type of x1 and x2", type(x1), type(x2))

  def attention(x1, x2, dropout=True):
    
    # img = tf.nn.tanh(tf.compat.v1.layers.dense(image_tensor , out_dim))
    
    # ques = tf.nn.tanh(tf.compat.v1.layers.dense(question_tensor , out_dim))
    
    # ques = tf.expand_dims(ques , axis = -2)
    
    IQ = tf.nn.tanh(x1 + x2)

    print("IQ shpe: ", IQ.shape)
     
    if dropout:
        IQ = tf.nn.dropout(IQ , rate=1 - (0.5))
    
    temp = Dense(1)(IQ)
    #temp = Dropout(0.5)(temp)
    temp = tf.reshape(temp , [-1,temp.shape[1]])
    print("temp shpe: ", temp.shape)
    
    p = tf.nn.softmax(temp)
    print("p shpe: ", p.shape)
    
    p_exp = tf.expand_dims(p , axis = -1)
    
    att_layer = tf.reduce_sum(input_tensor=p_exp * x1 , axis = 1)
    print("attlay shpe: ", att_layer.shape)
    
    final_out = att_layer + x2
        
    return p , final_out

  att_l1 , att = attention(x1, x2)
    
  att_l2 , att = attention(x1, att)

  att_l3 , att = attention(x1, att)

  att_l4 , att = attention(x1, att)
    
  att = tf.nn.dropout(att , rate=1 - (0.4))
    
  # att = Dense(768)(att)
  # print("att shpe: ", att.shape)
    
  # att = tf.nn.softmax(att)
  print("att shpe: ", att.shape)
    
  print(att.shape)
    
  attention_layers = [att_l1 , att_l2, att_l3,att_l4]

  # Merge -> output
  # out = Multiply()([x1, x2])
  out = Dense(512, activation='tanh')(att)
  out = Dropout(0.6)(out)
  out = BatchNormalization(axis=1)(out)
  out = Dense(num_answers, activation='softmax')(out)
  print("out shape: ", out.shape)
  print("im_input shpe: ", im_input.shape)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile(Adam(lr=4e-4), loss='categorical_crossentropy', metrics=['accuracy'])

  return model


