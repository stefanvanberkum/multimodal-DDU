""" Simple visualizations on 2D toy datasets
    Dataset-generation is taken from code on the practical about Uncertainty Estimation
    which can be found at https://gits-15.sys.kth.se/dd2412-deep-learning-advanced/Simple-and-Effective-Methods-for-Uncertainty-Estimation 
"""
import tensorflow as tf
import numpy as np
from vis_models import fc_net, fc_resnet, fc_ensemble
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy 
from uncertainty import DDU, DDU_KD, DDU_VI
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


def one_hot_enc(y, K):
    # y_one_hot = np.zeros(np.shape(y)[0], K)
    return np.array(y[:, None] == np.arange(K))

def ood_data(num_samples = 200):
    X = np.zeros((num_samples, 2))
    X[:,0] = np.random.uniform(low=-1.5, high=-1.0, size=(num_samples,))
    X[:,1] = np.random.uniform(low=1.0, high=1.5, size=(num_samples,))
    return X

def generate_datasets(N,K,noise):
    """
    Function for creating datasets, taken from practical and modified for our purposes

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
    X_ood = ood_data()

    if(mode=='ddu'):
        features_train = encoder.predict(X)
        print("Features shape: ", np.shape(features_train))
        ddu = DDU(features_train, y)
        features_xy = encoder.predict(xy)
        features_ood = encoder.predict(X_ood)
        print("Features ood: ", np.shape(features_ood))
        # project to lower dimensions
        pca = TSNE(n_components=2)
        features_all = np.concatenate([features_train, features_ood], axis=0)
        print("Features_all: ", np.shape(features_all))
        y_ood = 3*np.ones((np.shape(X_ood)[0], ), dtype='uint8')
        y_all = np.concatenate([y, y_ood], axis=0)
        print("y_all: ", np.shape(y_all))

        # fig, ax = plt.subplots(1,1)

        features_all_projected = pca.fit_transform(features_all)

        x_min_proj = np.min(features_all_projected[:,0])
        x_max_proj = np.max(features_all_projected[:,0])
        y_min_proj = np.min(features_all_projected[:,1])
        y_max_proj = np.max(features_all_projected[:,1])

        xs_proj = np.linspace(x_min_proj, x_max_proj, res)
        ys_proj = np.linspace(y_min_proj, y_max_proj, res)
        xy_proj = np.asarray([(_x,_y) for _x in xs_proj for _y in ys_proj])
        predictions_all = model.predict(xy_proj)
        features_proj = encoder.predict(xy_proj)
        _, epistemic_all = ddu.predict(features_proj, softmax(predictions_all, axis=1))
        log_density_all = -epistemic_all
        _, train_epistemic = ddu.predict(features_train, softmax(model.predict(X), axis=1))
        log_train_density = -train_epistemic
        train_min_density = np.min(log_train_density)
        Z_proj = np.zeros((N,M))
        indices_proj = np.unravel_index(np.arange(num_samples), (N,M))
        Z_proj[indices_proj] = log_density_all-train_min_density
        z_min = Z_proj[np.isfinite(Z)].min()
        z_max = Z_proj[np.isfinite(Z)].max()
        Z_proj[Z_proj == np.inf] = z_max
        Z_proj[Z_proj == -np.inf] = z_min

        fig, ax = plt.subplots(1,1)
        ax.contourf(xs_proj, ys_proj, Z_proj.T,levels=50, linewidths=0.2, cmap='cividis')
        ax.scatter(features_all_projected[:, 0], features_all_projected[:,1], c=y_all, s=10, alpha =0.3,cmap=ListedColormap(['orange', 'blue', 'green', 'red']))
        # plt.colorbar()
        aleatoric, epistemic = ddu.predict(features_xy, softmax(predictions, axis=1))
        log_density = -epistemic
        # _, train_epistemic = ddu.predict(features_train, softmax(model.predict(X), axis=1))
        # log_train_density = -train_epistemic
        # train_min_density = np.min(log_train_density)
        Z = np.clip(Z, -1, z_max) / z_max
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
    ax.scatter(X_ood[:,0], X_ood[:, 1], c='red', s=10, alpha = 0.3)
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
    model, encoder = fc_resnet(in_shape=(2,), num_classes=3)
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