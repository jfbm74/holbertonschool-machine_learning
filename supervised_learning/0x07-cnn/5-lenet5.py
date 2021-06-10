#!/usr/bin/env python3
"""Function that builds a modified version of
the LeNet-5 architecture using tensorflow"""

import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(X)
    pool1 = K.layers.MaxPool2D((2, 2), (2, 2))(conv1)
    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                            kernel_initializer='he_normal',
                            activation='relu')(pool1)
    pool2 = K.layers.MaxPool2D((2, 2), (2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    f1 = K.layers.Dense(units=120,
                        kernel_initializer='he_normal',
                        activation='relu')(flatten)
    f2 = K.layers.Dense(units=84, kernel_initializer='he_normal',
                        activation='relu')(f1)
    f3 = K.layers.Dense(units=10, kernel_initializer='he_normal',
                        activation='softmax')(f2)
    optimizer = K.optimizers.Adam()
    model = K.Model(inputs=X, outputs=f3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
