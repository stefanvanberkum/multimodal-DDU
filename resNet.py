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
from scipy.stats import entropy
from scipy.special import softmax, logsumexp
from spectral_normalization import spectral_normalization

def resnet_uncertainty(y, mode='softmax'):
    """Calculates simple uncertainty measures for single (deterministic) resnet
    :param y: output-logits of shape (n_obs, n_classes)
    :param mode: mode for uncertainty ('softmax' or 'energy')
    """
    if(mode=='softmax'):
        # use softmax entropy as uncertainty
        probs = softmax(y, axis=-1)
        aleatoric = entropy(probs, axis=-1)
        epistemic = aleatoric
    elif(mode=='energy'):
        # aleatoric: softmax entropy, epistemic: unnormalized softmax density (logsumexp of logits)
        probs = softmax(y, axis=-1)
        aleatoric = entropy(probs, axis=-1)
        epistemic = -logsumexp(y, axis=-1)
    else:
        aleatoric = 0
        epistemic = 0
    
    return aleatoric, epistemic

def resnet(stages, N, in_filters, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, use_bottleneck=False, ablate=False, coeff=3.0):
    """ResNet as described by He et al. (2015) with changes described by Mukhoti et al. (2023)

    :param stages: list of number of filters
    :param N: Number of blocks per stage.
    :param in_filters: Number of filters for input convolution
    :param k: Widening factor.
    :param in_shape: Input shape.
    :param n_out: Output size.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param modBlock: variable deciding if modified blocks are used (Leaky-ReLU, modified skip-connections etc.)
    :param ablate: variable deciding if ablation study, if true normal skip-connections are used with other modifications
    """
    

    inputs = Input(shape=in_shape)


    # First convolution
    if(modBlock):
        x = spectral_normalization(Conv2D(in_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)), coeff=3.0)(inputs)
        #
        # conv = Conv2D(in_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
       #x = wrapped_conv_layer(conv, input_size=in_shape,in_c = 3,coeff=3.0)
    else: 
        x = Conv2D(in_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
    

    # Add ResNet blocks.
    for i in range(len(stages)):
        n_filters = stages[i]
        for j in range(N):
            if i > 0 and j == 0:
                if(not use_bottleneck):
                    x = block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock, ablate=ablate)
                else: 
                    x = bottleneck_block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock, ablate=ablate)
            else:
                if(not use_bottleneck):
                    x = block(x, n_filters, dropout, weight_decay, modBlock=modBlock, ablate=ablate)
                else:
                    x = bottleneck_block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock, ablate=ablate)
    # Pooling and dense output layer with softmax activation.
    x = BatchNormalization()(x)
    x = relu(x)
    x_pool = GlobalAveragePooling2D()(x)
    outputs = Dense(n_out, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x_pool)

    model = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs, x_pool)

    # TODO: This seems to depend on the model -> move out of WRN and into main.
    # Compile model.
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.Accuracy()
    if(modBlock):
        runEagerly = True
    else: 
        runEagerly=False
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'], run_eagerly=runEagerly)

    return model, encoder



def block(x, n_filters, dropout, weight_decay, downsample=False, modBlock = True, ablate=False, coeff=3.0):
    """Basic ResNet block.

    :param x: Input.
    :param n_filters: Number of filters.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param downsample: True if the layer should downsample.
    """

    if(modBlock): 
        activation = tf.keras.layers.LeakyReLU(alpha = 0.01)
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
        x = spectral_normalization(Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)), coeff=coeff)(x)
    else: 
        x = Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = activation(x)
    if(dropout > 0):
        x = Dropout(dropout)(x)
    
    if(modBlock):
        x = spectral_normalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)), coeff=3.0)(x)
    else:
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    

    if downsample:
        if(not modBlock or ablate):
            x_skip = Conv2D(n_filters, 1, 2, use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x_skip)
        else: 
            # change to standard ResNet: Authors use strided average pooling instead of a 1x1-conv. layer
            # see: Appedix C.1 - "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
            # x_skip = AveragePooling2D(pool_size =  (start_stride, start_stride), strides=start_stride)(x)

            # # add padded zero entries if number of channels increases
            # if(n_filters > x_skip.shape[-1]):
            #     zero_entries = tf.zeros(shape=[x_skip.shape[0], x_skip.shape[1], x_skip.shape[2], n_filters-x_skip.shape[3]])

            #     # concatenate in dimension of  channels
            #     x_skip = tf.concat([x_skip, zero_entries], axis=-1)

            if(x_skip.shape[1] % 2 == 0):
                # print("Downsample")
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
                # print("Shape x_skip: ", tf.shape(x_skip))
    elif(n_filters > x_skip.shape[-1]):
        # print("INcrease depth")
        missing = n_filters-x_skip.shape[3]
        x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])

    x = Add()([x_skip, x])

    return x


def bottleneck_block(x, n_filters, dropout, weight_decay, downsample=False, modBlock = True, ablate=False):
    """Basic ResNet-bottleneck-block.

    :param x: Input.
    :param n_filters: Number of filters.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param downsample: True if the layer should downsample.
    """

    if(modBlock): 
        activation = tf.keras.layers.LeakyReLU(alpha = 0.01)
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
        x = spectral_normalization(Conv2D(n_filters, 1, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 1, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)


    x = BatchNormalization()(x)
    x = activation(x)

    if(modBlock):
        x = spectral_normalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization()(x)
    x = activation(x)
    if(dropout > 0):
        x = Dropout(dropout)(x)
    
    if(modBlock):
        x = spectral_normalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
        

    x = BatchNormalization()(x)
    x = activation(x)

    if(modBlock):
        x = spectral_normalization(Conv2D(4*n_filters, 1, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(4*n_filters, 1, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
        
    
    
    # print("Shape x: ", tf.shape(x))
    # print("N filters: ", n_filters)
    

    if downsample:
        if(not modBlock or ablate):
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
                # print("Downsample")
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
                # print("Shape x_skip: ", tf.shape(x_skip))
    elif(n_filters > x_skip.shape[-1]):
        # print("INcrease depth")
        missing = n_filters-x_skip.shape[3]
        x_skip = tf.pad(x_skip, [[0,0], [0,0], [0,0], [missing //2, -(missing // -2)]])

    x = Add()([x_skip, x])

    return x

        
    





