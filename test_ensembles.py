""" Test ensembling of wide-res-net and res-net"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import ensembles

model = "resnet" # 'resnet' or 'wide-res-net'
n_members = 5


def normalize(image, label):
    return tf.cast(image, tf.float32)/255., label

if __name__ == '__main__':

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ds_train = tfds.load('mnist', split='train')
    ds_test = tfds.load('mnist', split='test')

    if(model=='resnet'):
        ensemble = ensembles.ensemble_resnet(n_members, stages=[64,128],N=2,in_filters=64, in_shape=(28,28,1), n_out = 10, modBlock = True, ablate=True)
    elif(model == 'wide-res-net'):
        ensemble = ensembles.ensemble_wrn(n_members, N=4, in_shape=(28,28,1), k=10, n_out=10, modBlock=True)
    else:
        ensemble = ensembles.ensemble_resnet(n_members, stages=[64,128,256,512],N=2,in_filters=64, in_shape=(28,28,1), n_out = 10, modBlock = True)

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
    outputs = []
    for i in range(n_members):
        out = ensemble[i](image)
        outputs.append(out.numpy())
        # print("Outputs {}: {}".format(i,outputs[i]))
        # ensemble[i].summary()
    outputs = np.array(outputs)
    print("Outputs: ", outputs)
    print("Shape: ", np.shape(outputs))

    # test uncertainty
    aleatoric, epistemic = ensembles.ensemble_uncertainty(outputs, 'entropy')
    print("Aleatoric Uncertainty: ", aleatoric)
    print("Epistemic uncertainty: ", epistemic)


    # test training for one epoch
    for i in range(n_members):
        ensemble[i].fit(x=trainX, y=trainY,batch_size=128,epochs=1,validation_data=(testX, testY))