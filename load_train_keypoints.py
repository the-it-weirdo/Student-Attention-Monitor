# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:40:13 2023

@author: Yassine
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


# =================================================================================================================
ACCURACY_THRESHOLD = 0.9998


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %
                  (ACCURACY_THRESHOLD*100))
            self.model.stop_training = True


callbacks = myCallback()  # used to stop train at acc 99.98%
# ===================================================================================================================
# you do not need this function if your data is formated as [x1 y1 z1 x2 y2 z2 x3 y3 z3 ..... label]


def process_data(path):
    ldata = np.load(path, allow_pickle=1)
    # print(ldata)
    # ldata = ldata[~np.isnan(ldata)]

    data = []
    labels = []
    for dt in ldata:
        samp = np.reshape(dt[0], dt[0].shape[0]*3)

        if dt[1] == 'Engaged':
            labels.append(0)
        else:
            labels.append(1)

        data.append(samp)

    return np.array(data), np.array(labels, dtype=np.uint8)
# =================================================================================================================


def generate_model(input_sz, num_classes):
    inputs = Input(shape=input_sz)
    L1 = Dense(400, activation='relu')(inputs)
    L2 = Dense(200, activation='relu')(L1)
    L3 = Dense(100, activation='relu')(L2)
    L4 = Dense(50, activation='relu')(L3)
    L5 = Dense(32, activation='relu')(L4)
    L6 = Dense(12, activation='relu')(L5)

    L7 = Dense(num_classes, activation='softmax')(L6)

    cnn_model = Model(inputs=inputs, outputs=L7)

    cnn_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    cnn_model.summary()

    return cnn_model


keypoints, labels = process_data("Data.npy")

num_classes = 2

input_size = keypoints.shape[1]

tr_features = keypoints.reshape(-1, input_size, 1)
tr_labels = to_categorical(labels)

batch_size = 32
epochs = 350


engage_model = generate_model(input_size, num_classes)
engage_model.fit(tr_features, tr_labels, batch_size=batch_size,
                 epochs=epochs, callbacks=[callbacks], verbose=1)


model_filename = "engage_2class_1.h5"
engage_model.save(model_filename)
