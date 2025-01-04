#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    This code trains a convolutional network to find flaws in
    ultrasonic data. See https://arxiv.org/abs/1903.11399 for details.
'''

from __future__ import print_function
import keras
from keras import backend as K
from keras import Input, layers
from keras import Model

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.interactive(True)

from os import listdir
from os.path import isfile, join
import uuid
import time


custom_name = "ndt_nn_relu"


w,h = 256,256                       # initial data size
window = 7                          # window for the first max-pool operation

run_uuid = uuid.uuid4()             #unique identifier is generated for each run

path = "../../data/training/"  #training data path
vpath = "../../data/validation/"       #validation data path

'''     The data_generator reads raw binary UT data from the pre-processed files
        and preconditions it for ML training. '''
def data_generator( batch_size = 10):
    input_files = [f for f in listdir(path) if isfile(join( path,f)) and f.endswith('.bins') ]
    np.random.shuffle(input_files)          # we'll take random set from available data files
    input_files = input_files[0:100]        # limit to 100 files per epoch
    xs = np.empty( (0), dtype='float32')    #  input data
    ys = np.empty((0,2), dtype='float32')   #  label data
    for i in input_files:
        bxs = np.fromfile(path+i, dtype=np.uint16).astype('float32')
        bxs -= bxs.mean()
        bxs /= bxs.std() +0.00001           #avoid division by zero
        xs = np.concatenate((xs,bxs))
        bys = np.loadtxt(path + i[:-5] +'.labels')
        ys = np.concatenate((ys,bys) )

    xs = np.reshape(xs, (-1,256,256,1), 'C')

    rows = xs.shape[0]
    cursor = 0
    while True:
        start = cursor
        cursor += batch_size
        if(cursor > rows):
            cursor = 0
        bxs = xs[start:cursor,:,:,:]
        bys = ys[start:cursor,0]
        # yield( (xs[start:cursor,:,:,:], ys[start:cursor,0]) )
        yield (xs[start:cursor, :, :, :],
               {'binary_output': ys[start:cursor, 0],
                'regression_output': ys[start:cursor, 1]})

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Define the model
input_tensor = Input(shape=(w, h, 1))

# Start with max-pool to envelop the UT-data
ib = layers.MaxPooling2D(pool_size=(window, 1), padding='valid')(input_tensor)

# Build the network
cb = layers.Conv2D(96, 3, padding='same', activation='relu')(ib)
cb = layers.Conv2D(64, 3, padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D((2, 8), padding='same')(cb)

cb = layers.Conv2D(48, 3, padding='same', activation='relu')(cb)
cb = layers.Conv2D(32, 3, padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D((3, 4), padding='same')(cb)
cb = layers.Flatten()(cb)
cb = layers.Dense(14, activation='relu', name='RNN')(cb)

# Apply temperature scaling
iscrack = layers.Dense(1, activation='relu', name='binary_output')(cb)
regression_output = layers.Dense(1, activation='relu', name='regression_output')(cb)

# Define the model
model = Model(input_tensor, [iscrack, regression_output])

# Compile the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, clipnorm=1.0)
model.compile(
    optimizer='adam',
    loss={
        'binary_output': 'binary_crossentropy',
        'regression_output': 'mean_squared_error'
    },
    loss_weights={
        'binary_output': 1.0,  # Weight for the classification loss
        'regression_output': 10.0  # Higher weight for regression to emphasize it more
    }
)
model.summary()


test_uuid = "FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31"
rxs = np.fromfile(vpath+test_uuid+".bins", dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')
rys = np.loadtxt(vpath+test_uuid+".labels", dtype=np.float32)

validation_uuid = "FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31"
xs = np.fromfile(vpath+validation_uuid+".bins", dtype=np.uint16 ).astype('float32')
xs -= xs.mean()
xs /= xs.std()+0.0001
xs = np.reshape( xs, (-1,256,256,1), 'C')
ys = np.loadtxt(vpath+validation_uuid+".labels", dtype=np.float32)


class DebugCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predictions = model.predict(rxs)
        # Split predictions into binary and regression outputs
        binary_predictions = predictions[0]
        regression_predictions = predictions[1]
        # Combine ground truth and predictions
        res = np.column_stack((rys[:, 0], rys[:, 1], binary_predictions, regression_predictions))
        plt.scatter(res[:, 1], res[:, 3], color='blue', label='Regression Predictions')
        plt.scatter(res[:, 1], res[:, 2], color='red', label='Binary Predictions')
        plt.legend()
        plt.show()


debug = DebugCallback()

callbacks = [  keras.callbacks.TensorBoard(log_dir='../log', histogram_freq=1)
             , keras.callbacks.ModelCheckpoint('modelcpnt'+custom_name+'.keras', monitor='val_loss', verbose=1, save_best_only=True)
             , debug ]


model.fit(data_generator(100), epochs=100, validation_data = (xs, {'binary_output': ys[:, 0], 'regression_output': ys[:, 1]}), steps_per_epoch=60, callbacks=callbacks)

predictions = model.predict(rxs)
binary_predictions = predictions[0]
regression_predictions = predictions[1]
res = np.column_stack((rys[:, 0], rys[:, 1], binary_predictions, regression_predictions))

plt.scatter(res[:, 1], res[:, 3], color='blue', label='Regression Predictions')
plt.scatter(res[:, 1], res[:, 2], color='red', label='Binary Predictions')
plt.legend()
plt.show()
np.savetxt('results_' + custom_name+ '.txt', res)

