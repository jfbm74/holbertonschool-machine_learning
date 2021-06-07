#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent
    """
    listing = []
    if save_best:
        save_model = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        listing.append(save_model)

    def lr_scheduler(epoch):
        """Schedule learning rate """
        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        if learning_rate_decay:
            lr_decay = K.callbacks.LearningRateScheduler(
                lr_scheduler,
                verbose=1
            )
            listing.append(lr_decay)
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=patience
            )
            listing.append(early_stop)
    else:
        early_stop = None
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=listing,
        validation_data=validation_data,
        shuffle=shuffle,
    )
    return history
