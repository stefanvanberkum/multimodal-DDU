""" Models for simple visualizations on 2D datasets"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import SpectralNormalization, BatchNormalization

# fully-connected model
def fc_net(in_shape, num_classes, learning_rate=0.01, weight_decay=1e-04):
    inputs = Input(shape=in_shape)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    features = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    output = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(features) # outputs are logits

    model = Model(inputs=inputs, outputs=output)
    encoder = Model(inputs=inputs, outputs = features)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])

    return model, encoder

# fully-connected res-net model
def fc_resnet(in_shape, num_classes, sn=False, learning_rate = 0.01, weigth_decay=1e-4):
    inputs = Input(shape=in_shape)
    x = SpectralNormalization(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weigth_decay)))(inputs)
    x = BatchNormalization()(x)
    x_skip = x
    x = SpectralNormalization(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weigth_decay)))(x)
    x = BatchNormalization()(x)
    x = SpectralNormalization(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(weigth_decay)))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.activations.relu(x)
    features = SpectralNormalization(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weigth_decay)))(x)
    features = BatchNormalization()(features)
    output = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(weigth_decay))(features) # outputs are logits

    model = Model(inputs=inputs, outputs=output)
    encoder = Model(inputs=inputs, outputs = features)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=opt, loss=loss_func, metrics=[metric], run_eagerly=True)
    
    return model, encoder


def fc_ensemble(in_shape, num_classes, n_members, learning_rate=0.01, weight_decay=1e-04):
    models = []
    encoders = []
    for i in range(n_members):
        inputs = Input(shape=in_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        features = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(num_classes)(features) # outputs are logits

        member = Model(inputs=inputs, outputs=output)
        encoder_member = Model(inputs=inputs, outputs = features)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy()

        member.compile(optimizer=opt, loss=loss_func, metrics=[metric])

        models.append(member)
        encoders.append(encoder_member)

    return models, encoders
    