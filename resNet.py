"""
    ResNet implementation
The implementation follows the ResNet as described by He et al. (2015). with modifications described by Mukhoti et al. (2023)
"""

import tensorflow as tf
from tensorflow import pad
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, SpectralNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa



def resnet(stages, N, in_filters, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, use_bottleneck=False):
    """WRN-n-k as described by Zagoruyko and Komodakis (2017).

    This network has n=7N layers (2N for each of the three stages with an additional convolution at the start and at
    stage two and three for downsampling).

    Note that:
    - WRN-28-10 has N=4 and k=10.

    :param stages: list of number of filters
    :param N: Number of blocks per stage.
    :param in_filters: Number of filters for input convolution
    :param k: Widening factor.
    :param in_shape: Input shape.
    :param n_out: Output size.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    """
    
    inputs = Input(shape=in_shape)


    # First convolution
    if(modBlock):
        x = SpectralNormalization(Conv2D(in_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(inputs)
    else: 
        x = Conv2D(in_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
    

    # Add ResNet blocks.
    for i in range(len(stages)):
        n_filters = stages[i]
        for j in range(N):
            if i > 0 and j == 0:
                if(not use_bottleneck):
                    x = block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock)
                else: 
                    x = bottleneck_block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock)
            else:
                if(not use_bottleneck):
                    x = block(x, n_filters, dropout, weight_decay, modBlock=modBlock)
                else:
                    x = bottleneck_block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock)
    # Pooling and dense output layer with softmax activation.
    x = BatchNormalization()(x)
    x = relu(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(n_out, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=inputs, outputs=outputs)

    # TODO: This seems to depend on the model -> move out of WRN and into main.
    # Compile model.
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.Accuracy()
    if(modBlock):
        runEagerly = True
    else: 
        runEagerly=False
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=runEagerly)
    return model



def block(x, n_filters, dropout, weight_decay, downsample=False, modBlock = True):
    """Basic ResNet block.

    :param x: Input.
    :param n_filters: Number of filters.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param downsample: True if the layer should downsample.
    """

    if(modBlock): 
        activation = tf.keras.layers.LeakyReLU()
    else: 
        activation = tf.keras.layers.ReLU()


    if downsample:
        start_stride = 2
    else:
        start_stride = 1
    x_skip = x
    x = BatchNormalization()(x)
    x = activation(x)
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = activation(x)
    if(dropout > 0):
        x = Dropout(dropout)(x)
    
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    
    print("Shape x: ", tf.shape(x))
    print("N filters: ", n_filters)
    

    if downsample:
        if(not modBlock):
            x_skip = Conv2D(n_filters, 1, 2, use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x_skip)
        else: 
            # change to standard ResNet: Authors use strided average pooling instead of a 1x1-conv. layer
            # see: Appedix C.1 - "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
            # x_skip = AveragePooling2D(pool_size =  (start_stride, start_stride), strides=start_stride)(x)

            # # add padded zero entries if number of channels increases
            # if(n_filters > x_skip.shape[-1]):
            #     zero_entries = tf.zeros(shape=[x_skip.shape[0], x_skip.shape[1], x_skip.shape[2], n_filters-x_skip.shape[3]])

            #     # concatenate in dimension of channels
            #     x_skip = tf.concat([x_skip, zero_entries], axis=-1)

            if(x_skip.shape[1] % 2 == 0):
                print("Downsample")
                x_skip = AveragePooling2D(pool_size =  2, strides=2)(x_skip)
            else:
                #just 1x1 average pooling in that case, could be replaced with usual 1x1-convolutions
                x_skip = AveragePooling2D(pool_size = 1, strides=2)(x_skip)
            # add padded zero entries if number of channels increases
            if(n_filters > x_skip.shape[-1]):
                # zero_entries = tf.zeros(shape=[x_skip.shape[0], x_skip.shape[1], x_skip.shape[2], n_filters-x_skip.shape[3]])
                # concatenate in dimension of channels
                missing = n_filters-x_skip.shape[3]
                x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])
                print("Shape x_skip: ", tf.shape(x_skip))
    elif(n_filters > x_skip.shape[-1]):
        print("INcrease depth")
        missing = n_filters-x_skip.shape[3]
        x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])

    x = Add()([x_skip, x])

    return x


def bottleneck_block(x, n_filters, dropout, weight_decay, downsample=False, modBlock = True):
    """Basic ResNet-bottleneck-block.

    :param x: Input.
    :param n_filters: Number of filters.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param downsample: True if the layer should downsample.
    """

    if(modBlock): 
        activation = tf.keras.layers.LeakyReLU()
    else: 
        activation = tf.keras.layers.ReLU()


    if downsample:
        start_stride = 2
    else:
        start_stride = 1
    x_skip = x

    x = BatchNormalization()(x)
    x = activation(x)
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 1, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 1, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)


    x = BatchNormalization()(x)
    x = activation(x)

    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = activation(x)
    if(dropout > 0):
        x = Dropout(dropout)(x)
    
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
        

    x = BatchNormalization()(x)
    x = activation(x)

    if(modBlock):
        x = SpectralNormalization(Conv2D(4*n_filters, 1, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(4*n_filters, 1, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
        
    
    
    print("Shape x: ", tf.shape(x))
    print("N filters: ", n_filters)
    

    if downsample:
        if(not modBlock):
            x_skip = Conv2D(4*n_filters, 1, 2, use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x_skip)
        else: 
            # change to standard ResNet: Authors use strided average pooling instead of a 1x1-conv. layer
            # see: Appedix C.1 - "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
            # x_skip = AveragePooling2D(pool_size =  (start_stride, start_stride), strides=start_stride)(x)

            # # add padded zero entries if number of channels increases
            # if(n_filters > x_skip.shape[-1]):
            #     zero_entries = tf.zeros(shape=[x_skip.shape[0], x_skip.shape[1], x_skip.shape[2], n_filters-x_skip.shape[3]])

            #     # concatenate in dimension of channels
            #     x_skip = tf.concat([x_skip, zero_entries], axis=-1)

            if(x_skip.shape[1] % 2 == 0):
                print("Downsample")
                x_skip = AveragePooling2D(pool_size =  2, strides=2)(x_skip)
            else:
                #just 1x1 average pooling in that case, could be replaced with usual 1x1-convolutions
                x_skip = AveragePooling2D(pool_size = 1, strides=2)(x_skip)
            # add padded zero entries if number of channels increases
            if(4*n_filters > x_skip.shape[-1]):
                # zero_entries = tf.zeros(shape=[x_skip.shape[0], x_skip.shape[1], x_skip.shape[2], n_filters-x_skip.shape[3]])
                # concatenate in dimension of channels
                missing = 4*n_filters-x_skip.shape[3]
                x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])
                print("Shape x_skip: ", tf.shape(x_skip))
    elif(n_filters > x_skip.shape[-1]):
        print("INcrease depth")
        missing = n_filters-x_skip.shape[3]
        x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])

    x = Add()([x_skip, x])

    return x

        
    





