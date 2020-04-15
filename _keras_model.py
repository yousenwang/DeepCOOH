"""
Auther: Ethan Wang
Started Date: 04/15/2020
Email: yousenwang@gmail.com
"""

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import keras
import tensorflow as tf

def build_model(p):
  model = Sequential([
    Dense(64, activation='relu', input_shape=[p]),
    Dense(64, activation='relu'),
    Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
      print(f'{epoch}\n  ')
    print('.', end='')