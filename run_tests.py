"""
Script for running the tests described in the paper
     
    "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)

to replicate Table 1.
- Architectures: Wide-Res-Net 28-10, SNGP, Deep Ensemble (5 Ensemble members)
- Datasets: CiFAR-10, CIFAR-100, SVHN, Tiny-ImageNet
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np 
from WRN import WRN
from resNet import resnet

train_ds_name = "cifar-10"


# load training datasets involved
if(train_ds_name == "cifar-10"):
    pass
elif(train_ds_name == "cifar-100"):
    pass

# load pre-trained models


# load test datasets




# 



