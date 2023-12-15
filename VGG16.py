"""
Script for the VGG-16 model based on "Very Deep Convolutional Networks for Large-Scale Image Recognition", Simonyan and Zisserman (2015)
Adjusted to match the implementation in "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
"""

from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPool2D , Flatten, Input, BatchNormalization, Activation, SpectralNormalization
import tensorflow as tf

def vgg_16(in_shape, stages=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], kernel_size = (3,3), padding = "same", modBlock = False, ablate = False, n_out = 10):

    """
    Builds the VGG-16 model
    Returns the model and the encoder
    """

    input = Input(shape = in_shape)

    if modBlock:
        activation = "leaky-relu"
    else:
        activation = "relu"

    if modBlock: 
        x = SpectralNormalization(Conv2D(input_shape = in_shape, filters = stages[0], kernel_size = kernel_size, padding = padding, activation = None))(input)
    else:
        x = Conv2D(input_shape = in_shape, filters = stages[0], kernel_size = kernel_size, padding = padding, activation = None)(input)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    i = 2
    for filters in stages[1:]:
        if modBlock: 
            x = SpectralNormalization(Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, activation = None))(x)
        else:
            x = Conv2D(filters = filters, kernel_size=kernel_size, padding = padding, activation = None)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        if i in [4, 7, 10, 13]:
            x = MaxPool2D(pool_size = (2,2), strides = (2,2))(x)
        i+=1

    x = AveragePooling2D(pool_size = (1,1), strides = (1,1))(x)
    x = Flatten()(x)

    encoder = Model(inputs = input, outputs = x)
    encoder.summary()

    output = (Dense(units = n_out))(x) # 10 Classes for mnist

    model = Model(inputs = input, outputs = output)
    
    opt = tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    if(modBlock):
        runEagerly = True
    else: 
        runEagerly= False

    model.compile(optimizer = opt, loss = loss, metrics=['accuracy'], run_eagerly = runEagerly)
    model.summary()

    return model, encoder