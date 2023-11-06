""" Script for testing resnet implementation"""

import tensorflow as tf
import tensorflow_datasets as tfds
from resNet import resnet, resnet_uncertainty
import matplotlib.pyplot as plt
import numpy as np


# def normalize_img(image, label):
#         """Normalizes images: `uint8` -> `float32`."""
#     return tf.cast(image, tf.float32) / 255., label
def normalize(image, label):
    return tf.cast(image, tf.float32)/255., label

if __name__ == '__main__':

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ds_train = tfds.load('mnist', split='train')
    ds_test = tfds.load('mnist', split='test')

    # test on ResNet18 architecture
    resNet18 = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(28,28,1), n_out = 10, modBlock = True)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    # resNet18.compile(optimizer=tf.keras.optimizers.legacy.SGD(0.01, 0.9),loss=loss_func, metrics = [metric])

    
    # construct training and test data
    trainX = np.zeros((60000, 28,28,1), dtype=np.float32)
    trainY = np.zeros((60000,), dtype=np.int32)
    testX = np.zeros((10000, 28, 28,1), dtype=np.float32)
    testY = np.zeros((10000,), dtype=np.int32)
    for i, elem in enumerate(ds_train):
        # print(elem)
        trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
        trainY[i] = elem['label']
    for i, elem in enumerate(ds_test):
        # print(elem)
        testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
        testY[i] = elem['label']
    
    # show image
    image = trainX[0]
    plt.imshow(image)
    plt.show()

    # test model on single image - expand batch dim first

    image = np.concatenate((np.expand_dims(trainX[0],axis=0), np.expand_dims(trainX[1], axis=0)), axis=0)
    # image = tf.expand_dims(image, axis=0)
    out = resNet18(image)
    out = out.numpy()
    print("Output: ", out)
    print("Out shape: ", np.shape(out))
    aleatoric, epistemic = resnet_uncertainty(out, mode='energy')
    print("Aleatoric: ", aleatoric)
    print("Epistemic: ", epistemic)


    resNet18.fit(x=trainX, y=trainY,batch_size=128,epochs=10,validation_data=(testX, testY))
    
    