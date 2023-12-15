"""
Script for the LeNet model based on "Gradient-Based Learning Applied to Document Recognition", LeCun et al. (1998)
Adjusted to match the implementation in "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
"""

from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input
import tensorflow as tf

def lenet(in_shape, kernel_size = (5,5), padding = "same", activation="relu", modBlock = False, ablate = False, n_out = 10):
    """
    Builds the LeNet model
    Conv2D -> MaxPool2D -> Conv2D -> MaxPool2D -> Flatten -> Dense 120 -> Dense 84 -> Dense n_out
    Returns the model and the encoder
    """

    input = Input(shape = in_shape)
    x = Conv2D(input_shape = in_shape, filters=6, kernel_size = kernel_size, padding = padding, activation = activation)(input)
    x = MaxPool2D(pool_size = (2,2), strides = (2,2))(x)

    x = Conv2D(filters = 16,kernel_size = kernel_size,padding = "valid", activation = activation)(x)
    x = MaxPool2D(2)(x)

    x = Flatten()(x)

    x = Dense(units = 120, activation = activation)(x)
    x = Dense(units = 84, activation = activation)(x)

    encoder = Model(inputs = input, outputs = x)
    encoder.summary()

    output = (Dense(units = n_out))(x) # 10 Classes for MNIST

    model = Model(inputs=input, outputs =output)
    
    opt = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9) #Set learning rate to 0.01, no info in the paper other than 0.1 learning rate --> No convergence without batch norm
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    if(modBlock):
        runEagerly = True
    else: 
        runEagerly=False

    model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'], run_eagerly = runEagerly)
    model.summary()

    return model, encoder