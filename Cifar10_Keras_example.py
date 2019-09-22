# -*- coding: utf-8 -*-


import numpy as np
import keras
from keras.layers import Conv2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Reshape, Permute, multiply, MaxPooling2D, AveragePooling2D
import albumentations as albu
from albumentations import (HorizontalFlip, ShiftScaleRotate, GridDistortion)
import keras.backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SENet import SEBlock


#Load cifar10 data
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.25, random_state = 2019)

# Swish activation function : https://arxiv.org/abs/1710.05941
def _swish(x):
  return K.sigmoid(x)*x

#helper function for convolution -> batch_normalization-> activation
def _conv_bn_act(filters = 32, kernel_size = (3,3), strides = 1, activation = "relu"):

  def f(input_x):

    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = (strides, strides), kernel_initializer="he_normal")(input_x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

  return f

#helper function for fully-connected -> batch_normalization-> activation
def _fc_bn_act(units, activation = "relu"):

  def f(input_x):

    x = Dense(units = units, kernel_initializer="he_normal")(input_x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x
  return f



def CNN(input_shape = (32,32,3), output_units = 10, activation = _swish, SE_flag = False, data_format = 'channels_last'):

  input_layer = keras.layers.Input(shape = input_shape)

  x = _conv_bn_act(filters=32, activation = activation)(input_layer)
  x = SEBlock(se_ratio=1, activation = activation, data_format=data_format)(x) if SE_flag == True else x
  x = _conv_bn_act(filters=32, activation = activation)(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = _conv_bn_act(filters=64, activation = activation)(x)
  x = SEBlock(se_ratio=1, activation = activation, data_format=data_format)(x) if SE_flag == True else x
  x = _conv_bn_act(filters=64, activation = activation)(x)
  x = Dropout(0.25)(x)

  #Use GlobalAveragePooling2D to replace flatten
  x = GlobalAveragePooling2D()(x)
  x = Reshape(1,1,x._keras_shape[1])(x) if data_format == 'channels_first' else x
    
  x = _fc_bn_act(units=256, activation = activation)(x)
  x = Dropout(0.25)(x)
  x = _fc_bn_act(units=128, activation = activation)(x)
  
  output_layer = Dense(units = output_units, activation="softmax", kernel_initializer="he_normal")(x)

  model = keras.models.Model(inputs = [input_layer], outputs = [output_layer])
  return model

batch_size = 128
lr = 1e-3
epochs = 20
optimizer = keras.optimizers.Adam(lr=lr)
loss = "categorical_crossentropy"
metric = ['accuracy']

steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_test) // batch_size

def input_generator(x,y,aug,batch_size):

  x_len = len(x)
  batch_x, batch_y = [],[]
  while True:

    batch_indices = np.random.choice(x_len, size = batch_size)
    
    for idx in batch_indices:
      batch_y.append(y[idx])
      batch_x.append(aug(image = x[idx])['image']/255.0)

    batch_x, batch_y = np.stack(batch_x), np.stack(batch_y)
    yield batch_x, batch_y
    batch_x, batch_y = [],[]
  
aug_for_train = albu.Compose([HorizontalFlip(p=0.5),
                              ShiftScaleRotate(shift_limit=0.1,scale_limit=0.25,rotate_limit=20,p=0.5),
                              GridDistortion(p=0.5)])
aug_for_valid = albu.Compose([])

train_gen = input_generator(x_train, y_train, aug_for_train, batch_size)
valid_gen = input_generator(x_test, y_test, aug_for_valid, batch_size)

def display_training_result(history):

  plt.figure(figsize=(16,12))

  plt.subplot(2,1,1)
  plt.plot(history.history['loss'], label = 'train_loss', color = 'g')
  plt.plot(history.history['val_loss'], label = 'valid_loss', color = 'r')
  plt.title('training/validation loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()

  plt.subplot(2,1,2)
  plt.plot(history.history['acc'], label = 'train_acc', color = 'g')
  plt.plot(history.history['val_acc'], label = 'valid_acc', color = 'r')
  plt.title('training/validation accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend()

  plt.show()

# Build the regular Convolution neural network
regular_cnn = CNN()
regular_cnn.compile(loss = loss, metrics = metric, optimizer = optimizer)
regular_cnn.summary()

regular_cnn_history = regular_cnn.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                                                validation_data=valid_gen, validation_steps=validation_steps)

# Build the convolution neural network with SE block in it
SE_cnn = CNN(SE_flag = True)
SE_cnn.compile(loss = loss, metrics = metric, optimizer = optimizer)
SE_cnn.summary()

SE_cnn_history = SE_cnn.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                                                validation_data=valid_gen, validation_steps=validation_steps)

display_training_result(regular_cnn_history)

display_training_result(SE_cnn_history)

regular_cnn_scores = regular_cnn.evaluate(x=x_test/255.0, y=y_test)
SE_cnn_scores = SE_cnn.evaluate(x=x_test/255.0, y=y_test)

print(f' Regular CNN :  loss {regular_cnn_scores[0]}, accuracy {regular_cnn_scores[1]}')
print(f' SE CNN : loss {SE_cnn_scores[0]}, accruacy {SE_cnn_scores[1]}')

