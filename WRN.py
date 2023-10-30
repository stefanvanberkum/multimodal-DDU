"""Wide residual network (WRN) implementation.

The implementation follows the WRN as described by Zagoruyko and Komodakis (2017).
"""

from tensorflow import pad
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


def WRN(N, k, in_shape, n_out, dropout=0, weight_decay=1e-4):
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
    """

    stages = [16 * k, 32 * k, 64 * k]
    inputs = Input(shape=in_shape)

    # First convolution.
    x = Conv2D(16, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)

    # Add ResNet blocks.
    for i in range(len(stages)):
        n_filters = stages[i]
        for j in range(N):
            if i > 0 and j == 0:
                x = block(x, n_filters, dropout, weight_decay, downsample=True)
            else:
                x = block(x, n_filters, dropout, weight_decay)

    # Pooling and dense output layer with softmax activation.
    x = BatchNormalization()(x)
    x = relu(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(n_out, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=inputs, outputs=outputs)

    # TODO: This seems to depend on the model -> move out of WRN and into main.
    # Compile model.
    opt = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def block(x, n_filters, dropout, weight_decay, downsample=False):
    """Basic ResNet block.

    :param x: Input.
    :param n_filters: Number of filters.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    :param downsample: True if the layer should downsample.
    """

    if downsample:
        start_stride = 2
    else:
        start_stride = 1

    x_skip = x

    x = BatchNormalization()(x)
    x = relu(x)
    x = Conv2D(n_filters, 3, start_stride, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Dropout(dropout)(x)
    x = Conv2D(n_filters, 3, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)

    if downsample:
        x_skip = Conv2D(n_filters, 1, 2, use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x_skip)

    x = Add()([x_skip, x])

    return x


