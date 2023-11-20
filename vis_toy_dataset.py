"""Simple visualizations on 2D toy datasets"""
import tensorflow as tf
import numpy as np
from vis_models import fc_net, fc_resnet, fc_ensemble
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy 
from uncertainty import DDU
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


def one_hot_enc(y, K):
    # y_one_hot = np.zeros(np.shape(y)[0], K)
    return np.array(y[:, None] == np.arange(K))

def generate_datasets(N,K,noise):
    """
    Function for creating datasets, taken and modified from practical
    """
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels

    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1.,N) # radius
        t = np.linspace(j*8,(j+1)*8,N) + np.random.randn(N)*noise * (r+1.0) # theta
        print(j, np.amin(t), np.amax(t))
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    X = np.array(X)
    y = np.array(y)
    y_onehot = one_hot_enc(y, K)
    return X, y, y_onehot

def visualize_predictions(X, y, model,encoder, min=-2.0, max=2.0, res=200, num_nets=1, mode='softmax'):
    xs = np.linspace(min, max, res)
    ys = np.linspace(min, max, res)
    N, M = len(xs), len(ys)
    xy = np.asarray([(_x,_y) for _x in xs for _y in ys])
    num_samples = xy.shape[0]
    print("NUm samples: ", num_samples)
    # predictions =  [batched_predict(params, xy)[1] for params in params_list]
    predictions = model.predict(xy)
    # predictions_ensemble = np.mean(predictions, axis=0)
    # total, data = js_terms(predictions)
    softmax_entropy = entropy(softmax(predictions, axis=1), axis=-1)

    Z, Z2, Z3 = np.zeros((N,M)), np.zeros((N,M)), np.zeros((N,M))
    indices = np.unravel_index(np.arange(num_samples), (N,M))
    print("indices: ", np.shape(indices))
    print("Shape Z: ", np.shape(Z))
    print("Shape predictions: ", np.shape(predictions))
    Z[indices] =np.argmax(softmax(predictions, axis=1), axis=1)


    if(mode=='ddu'):
        features_train = encoder.predict(X)
        print("Features shape: ", np.shape(features_train))
        isomap = Isomap(n_components=2)
        features_project = isomap.fit_transform(features_train)
        plt.scatter(features_project[:, 0], features_project[:, 1], c=y)
        plt.show()
        ddu = DDU(features_train, y)
        features_xy = encoder.predict(xy)
        aleatoric, epistemic  = ddu.predict(features_xy,softmax(predictions, axis=1))
        Z2[indices] = epistemic
        Z3[indices] = softmax_entropy
    else: 
        Z2[indices] = softmax_entropy
        Z3[indices] = softmax_entropy

    fig, axes = plt.subplots(2,2, figsize=(10,10))
    axes = axes.flatten()
    fig.tight_layout()

    # axes[0].scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    axes[1].contourf(xs, ys, Z.T, cmap=plt.cm.Spectral, levels=50)
    axes[0].contourf(xs, ys, Z3.T,levels=50)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    axes[3].contourf(xs, ys, Z2.T, cmap='magma', levels=50)

    axes[0].set_xlim([min, max]); axes[0].set_ylim([min, max]); 

    axes[0].title.set_text('Dataset')
    axes[1].title.set_text('Mean')
    axes[2].title.set_text('Data Uncertainty')
    axes[3].title.set_text('Knowledge Uncertainty')
    plt.show()

def visualize_single_uncertainty(X, y, model,encoder, min=-2.0, max=2.0, res=200, num_nets=1, mode='softmax'):
    xs = np.linspace(min, max, res)
    ys = np.linspace(min, max, res)
    N, M = len(xs), len(ys)
    xy = np.asarray([(_x,_y) for _x in xs for _y in ys])
    num_samples = xy.shape[0]
    predictions = model.predict(xy)
    # predictions_ensemble = np.mean(predictions, axis=0)
    # total, data = js_terms(predictions)
    softmax_entropy = entropy(softmax(predictions, axis=1), axis=-1)
    Z = np.zeros((N,M))

    indices = np.unravel_index(np.arange(num_samples), (N,M))

    if(mode=='ddu'):
        features_train = encoder.predict(X)
        print("Features shape: ", np.shape(features_train))
        ddu = DDU(features_train, y)
        features_xy = encoder.predict(xy)
        # project to lower dimensions
        pca = PCA(n_components=2)
        features_train_projected = pca.fit_transform(features_train)
        plt.scatter(features_train_projected[:, 0], features_train_projected[:,1], c=y, s=10, alpha =0.3,
                    cmap=ListedColormap(['orange', 'blue', 'green']))
        aleatoric, epistemic = ddu.predict(features_xy, softmax(predictions, axis=1))
        log_density = -epistemic

        _, train_epistemic = ddu.predict(features_train, softmax(model.predict(X), axis=1))
        log_train_density = -train_epistemic
        train_min_density = np.min(log_train_density)
        Z[indices] = log_density - train_min_density

        z_min = Z[np.isfinite(Z)].min()
        z_max = Z[np.isfinite(Z)].max()
        Z[Z == np.inf] = z_max
        Z[Z == -np.inf] = z_min

        Z = np.clip(Z, -1, z_max) / z_max
    else: 
        Z[indices] = softmax_entropy
    fig, ax = plt.subplots(1,1)
    ax.contourf(xs, ys, Z.T,levels=50, cmap='cividis')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha = 0.3, cmap=ListedColormap(['orange', 'blue', 'green']))#cmap=plt.cm.Spectral, alpha =0.7)
    plt.show()

def visualize_ensemble_uncertainty(X, y, models, min=-2.0, max=2.0, res=200, num_nets=1, mode='softmax'):
    xs = np.linspace(min, max, res)
    ys = np.linspace(min, max, res)
    N, M = len(xs), len(ys)
    xy = np.asarray([(_x,_y) for _x in xs for _y in ys])
    num_samples = xy.shape[0]
    predictions = [model.predict(xy) for model in models]
    # print("Shape prediction: ", np.shape(predictions))
    ensemble_predictions = np.mean(predictions, axis=0)
    softmax_entropy = entropy(softmax(ensemble_predictions, axis=1), axis=-1)
    # print("Shape entropy: ", np.shape(softmax_entropy))
    Z = np.zeros((N,M))
    indices = np.unravel_index(np.arange(num_samples), (N,M))
    if(mode=='softmax'):
        Z[indices] = softmax_entropy
    elif(mode =='mi'):
        # mutual information between parameters and labels captures epistemic uncertainty
        average_entropy = np.mean(entropy(softmax(predictions, axis=-1), axis=-1), axis=0)
        # Z[indices] = softmax_entropy-average_entropy
        Z[indices] = softmax_entropy-average_entropy
    elif(mode=='aleatoric'):
        average_entropy = np.mean(entropy(softmax(predictions, axis=-1), axis=-1), axis=0)
        Z[indices] = average_entropy
    else: 
        Z[indices] = softmax_entropy
    fig, ax = plt.subplots(1,1)
    ax.contourf(xs, ys, Z.T,levels=50, cmap='cividis')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha = 0.3, cmap=ListedColormap(['orange', 'blue', 'green']))#cmap=plt.cm.Spectral, alpha =0.7)
    plt.show()

if(__name__=="__main__"):
    N = 1000 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes

    np.random.seed(0)
    X, y, y_onehot = generate_datasets(N, K, noise=0.3) # default: 0.3
    # y = np.reshape(y, (-1, 1))
    # print("Y:", y)
    # print("Training X:", X.shape)
    # print("Training y:", y.shape)

    # train models
    model, encoder = fc_net(in_shape=(2,), num_classes=3)
    # print("X shape: ", np.shape(X))
    # logits = model(X[0:10])
    # print("Logits: ", logits)
    # model.summary()
    # train model on toy dataset
    # learning rate scheduler
    def scheduler(epoch, lr):
        if(epoch ==50 or epoch == 100):
            return lr/10
        else:
            return lr
  
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # train single model
    model.fit(x=X, y=y, epochs=150, batch_size = 64, callbacks=[lr_callback])


    # train ensemble
    # models, encodders = fc_ensemble(in_shape=(2,), num_classes=3, n_members = 3)
    # i= 0
    # for model in models:
    #     print("Training member %d..."%(i+1))
    #     model.fit(x=X, y=y, epochs=50, batch_size = 64, callbacks=[lr_callback])
    #     i+=1


    
    # visualize uncertainty
    visualize_single_uncertainty(X, y, model, encoder, mode='ddu')
    # visualize_ensemble_uncertainty(X,y,models, mode='dd')