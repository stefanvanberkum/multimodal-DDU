""" Code for ensembles of resnets and wide-res-nets"""
import tensorflow as tf
import numpy as np
from WRN import WRN
from resNet import resnet 
from tensorflow.keras import Input, Model
from scipy.stats import entropy
from scipy.special import softmax, logsumexp


def ensemble_resnet(n_members, stages, N, in_filters, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, use_bottleneck=False, ablate=False)->list:
    """Ensemble of ResNet's as described by He et al. (2015) with changes described by Mukhoti et al. (2023)

    :param n_members: number of ensemble members
    :param stages: list of number of filters
    :param N: Number of blocks per stage.
    :param in_filters: Number of filters for input convolution
    :param k: Widening factor.
    :param in_shape: Input shape.
    :param n_out: Output size.
    :param dropout: Dropout rate.
    :param weight_decay: Weight decay parameter.
    """

    # inputs = Input(shape=in_shape)
    ensemble = []

    for i in range(n_members):
        member,_ = resnet(stages, N, in_filters, in_shape, n_out, dropout, weight_decay, modBlock, use_bottleneck, ablate)
        ensemble.append(member)

    return ensemble
    

def ensemble_wrn(n_members, N, k, in_shape, n_out, dropout=0, weight_decay=1e-4, modBlock=True, ablate=False)-> list:
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
    ensemble = []
    for i in range(n_members):
        member,_ = WRN(N, k, in_shape, n_out, dropout, weight_decay, modBlock, ablate)
        ensemble.append(member)
    return ensemble


def ensemble_uncertainty(y, mode = 'entropy'):
    """ Calculates ensemble uncertainty
    :param y: ensemble outputs (logits) of shape (n_members, n_obs, n_classes)
    :param mode: mode for epistemic uncertainty (predictive entropy ('entropy') or mutual information ('mi'))
    :return: tupel containing (alteatoric, epistemic)
    """
    probs = softmax(y, axis=-1)
    if(mode == 'entropy'):
        # average probs over ensemble members
        avg_probs = np.mean(probs, axis=0)
        aleatoric = entropy(avg_probs, axis=1)
        epistemic = aleatoric
    elif(mode=='mi'):
        # average probs over ensemble members
        avg_probs = np.mean(probs, axis=0)
        aleatoric = entropy(avg_probs, axis=1)
        epistemic = aleatoric - np.mean(entropy(y, axis=-1), axis=0)
    else: 
        aleatoric = 0
        epistemic = 0

    return aleatoric, epistemic
