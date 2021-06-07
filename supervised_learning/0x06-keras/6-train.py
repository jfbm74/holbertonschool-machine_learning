#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent
    """
    if validation_data and early_stopping:
        early_stop = [K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)]
    else:
        early_stop = None
    output = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=early_stop,
        validation_data=validation_data,
        shuffle=shuffle,
    )
    return output
