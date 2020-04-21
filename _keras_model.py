"""
Auther: Ethan Wang
Started Date: 04/15/2020
Email: yousenwang@gmail.com
"""

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
def build_model(p):
  model = Sequential([
    Dense(64, activation='relu', input_shape=[p]),
    Dense(64, activation='relu'),
    Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                      initial_sparsity=0.0, final_sparsity=0.5,
                      begin_step=2000, end_step=4000)

  model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)

  # model.compile(loss='mse',
  #               optimizer=optimizer,
  #               metrics=['mae', 'mse'])
  # return model
  model_for_pruning.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model_for_pruning

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
      print(f'{epoch}\n  ')
    print('.', end='')