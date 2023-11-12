import tensorflow as tf
import tensorflow_datasets as tfds
from WRN import WRN, wrn_uncertainty
import matplotlib.pyplot as plt
import numpy as np


def normalize(image, label):
    return tf.cast(image, tf.float32)/255., label

if __name__ == '__main__':

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ds_train = tfds.load('mnist', split='train')
    ds_test = tfds.load('mnist', split='test')

    # test on ResNet18 architecture
    wideResNet = WRN(N=4, in_shape=(28,28,1), k=3, n_out=10, modBlock=False)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    
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
    image = tf.expand_dims(image, axis=0)
    out = wideResNet(image)
    out = out.numpy()
    print("Output: ", out)
    aleatoric, epistemic = wrn_uncertainty(out, mode='energy')
    print("Aleatoric: ", aleatoric)
    print("Epistemic: ", epistemic)


    # wideResNet.summary()

    wideResNet.fit(x=trainX, y=trainY,batch_size=128,epochs=10,validation_data=(testX, testY))

