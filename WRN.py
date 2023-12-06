"""Wide residual network (WRN) implementation.

The implementation follows the WRN as described by Zagoruyko and Komodakis (2017).
"""
import tensorflow as tf
from tensorflow import pad
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, SpectralNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from scipy.special import softmax, logsumexp
from scipy.stats import entropy


# class spectral_norm(tf.keras.layers.Wrapper):
#     def __init__(self, layer):
#         super(spectral_norm, self).__init__(layer)
#         self.layer = tfa.layers.SpectralNormalization(layer)
#     @tf.function
#     def call(self, inputs,training=None):
#         return self.layer(inputs)

class paddingLayer(tf.keras.layers.Layer):
    def __init__(self, numberOfPixels, mode="CONSTANT"):
        super(paddingLayer, self).__init__()
        self.numberOfPixels = numberOfPixels
        self.mode = mode
    def call(self, inputs, training=None, mask=None):
        if(training):
            outputs = tf.pad(inputs, tf.constant([[0, 0,], [self.numberOfPixels, self.numberOfPixels], [self.numberOfPixels, self.numberOfPixels], [0,0]]), mode=self.mode)
        else: 
            outputs = inputs
        return outputs

        
    



def WRN(N, k, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, ablate=False):
    """WRN-n-k as described by Zagoruyko and Komodakis (2017).

    This network has n=7N layers (2N for each of the three stages with an additional convolution at the start and at
    stage two and three for downsampling).

    Note that:
    - WRN-28-10 has N=4 and k=10.

    :param N: Number of blocks per stage.
    :param k: Widening factor.
    :param in_shape: Input shape.
    :param n_out: Output size.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param modBlock: variable deciding if modified blocks are used (Leaky-ReLU, modified skip-connections etc.)
    :param ablate: variable deciding if ablation study, if true normal skip-connections are used with other modifications
    """

    stages = [16 * k, 32 * k, 64 * k]
    inputs = Input(shape=in_shape)
    


    # First convolution
    if(modBlock):
        x = SpectralNormalization(Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(inputs)
    else: 
        x = Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
    

    # Add ResNet blocks.
    for i in range(len(stages)):
        n_filters = stages[i]
        for j in range(N):
            if i > 0 and j == 0:
                x = block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock, ablate=ablate)
            else:
                x = block(x, n_filters, dropout, weight_decay, modBlock=modBlock, ablate=ablate)

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


def WRN_with_augment(N, k, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, ablate=False, data_augment =False, batch_norm_momentum = 0.99, mean = [0.4914, 0.4822, 0.4465], variance=[0.2023**2, 0.1994**2, 0.2010**2]):
    """WRN-n-k as described by Zagoruyko and Komodakis (2017). Modified with preprocessing from https://github.com/omegafragger/DDU/blob/main/data/ood_detection/cifar10.py

    This network has n=7N layers (2N for each of the three stages with an additional convolution at the start and at
    stage two and three for downsampling).

    Note that:
    - WRN-28-10 has N=4 and k=10.

    :param N: Number of blocks per stage.
    :param k: Widening factor.
    :param in_shape: Input shape.
    :param n_out: Output size.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param modBlock: variable deciding if modified blocks are used (Leaky-ReLU, modified skip-connections etc.)
    :param ablate: variable deciding if ablation study, if true normal skip-connections are used with other modifications
    """

    stages = [16 * k, 32 * k, 64 * k]
    inputs = Input(shape=in_shape)
    

    
    # choice of data-augmentation methods is taken from https://github.com/omegafragger/DDU/blob/main/data/ood_detection/cifar100.py
    if(data_augment):
        # padding
        x = paddingLayer(numberOfPixels=4)(inputs)

        # random-crop
        x = tf.keras.layers.RandomCrop(height=32, width=32)(x)

        # random horizontal flip
        x = tf.keras.layers.RandomFlip(mode='horizontal')(x)

    # normalize on channel-wise mean and std (during training and inference)
    x = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)(x)


        


    # First convolution
    if(modBlock):
        x = SpectralNormalization(Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    

    # Add ResNet blocks.
    for i in range(len(stages)):
        n_filters = stages[i]
        for j in range(N):
            if i > 0 and j == 0:
                x = block(x, n_filters, dropout, weight_decay, downsample=True, modBlock=modBlock, ablate=ablate)
            else:
                x = block(x, n_filters, dropout, weight_decay, modBlock=modBlock, ablate=ablate)

    # Pooling and dense output layer with softmax activation.
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
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

def block(x, n_filters, dropout, weight_decay, downsample=False, modBlock = True, ablate=False, batch_norm_momentum=0.99):
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


    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = activation(x)
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else: 
        x = Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = activation(x)
    if(dropout > 0):
        x = Dropout(dropout)(x)
    
    if(modBlock):
        x = SpectralNormalization(Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay)))(x)
    else:
        x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    
    # print("Shape x: ", tf.shape(x))
    # print("N filters: ", n_filters)
    

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

            #     # concatenate in dimension of channels
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


def wrn_uncertainty(y, mode='softmax'):
    """Calculates simple uncertainty measures for single (deterministic) wide-resnet
    :param y: output-logits of shape (n_obs, n_classes)
    :param mode: mode for uncertainty ('softmax' or 'energy')
    """
    probs = softmax(y, axis=-1)
    if(mode=='softmax'):
        # use softmax entropy as uncertainty
        aleatoric = entropy(probs, axis=-1)
        epistemic = aleatoric
    elif(mode=='energy'):
        # aleatoric: softmax entropy, epistemic: unnormalized softmax density (logsumexp of logits)
        aleatoric = entropy(probs, axis=-1)
        epistemic = -logsumexp(y, axis=-1)
    else:
        aleatoric = 0
        epistemic = 0
    
    return aleatoric, epistemic


