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
import argparse
import tensorflow_probability as tfp
from WRN import WRN, wrn_uncertainty
from resNet import resnet, resnet_uncertainty
from ensembles import ensemble_resnet, ensemble_wrn, ensemble_uncertainty
from uncertainty import DDU, DDU_KD, DDU_CWKD, DDU_VI
from sklearn.metrics import roc_auc_score

# parameters for testing
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str) # 'wrn', 'resnet', 'wrn-ensemble', 'resnet-ensemble'
parser.add_argument("--train_ds", required=True, type=str) # 'cifar10', 'cifar100'
parser.add_argument("--test_ds", required=True, type=str) # 'cifar100', 'SVHN', 'Tiny-ImageNet'
parser.add_argument("--modBlock", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ablate", default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument("--n_epochs", default=350, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--test", default = "accuracy", type=str) # 'accuracy', 'ece', 'ood'
parser.add_argument("--n_runs", default = 10, type=int) # number of training runs to average over
parser.add_argument("--uncertainty", default='DDU', type=str) # 'DDU', 'energy', 'softmax'


# load pre-trained models
if(__name__ == "__main__"):
    args = parser.parse_args()
    test_model = args.model 
    train_ds = args.train_ds # 'cifar10', 'cifar100'
    test_ds = args.test_ds
    train_modBlock = args.modBlock
    train_ablate = args.ablate
    n_members = 5
    batch_size = args.batch_size
    test = args.test
    n_runs = args.n_runs
    uncertainty = args.uncertainty

    if(train_ds == 'cifar10'):
        n_classes = 10
        ds_train = tfds.load("cifar10", split='train')
        trainX = np.zeros((50000, 32, 32,3), dtype=np.float32)
        trainY = np.zeros((50000,), dtype=np.int32)

        for i, elem in enumerate(ds_train):
            # print(elem)
            trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            trainY[i] = elem['label']

    elif(train_ds == 'cifar100'):
        n_classes = 100
        ds_train = tfds.load('cifar100', split='train')
        trainX = np.zeros((50000, 32, 32,3), dtype=np.float32)
        trainY = np.zeros((50000,), dtype=np.int32)
        for i, elem in enumerate(ds_train):
            # print(elem)
            trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            trainY[i] = elem['label']

    if(test_ds == 'cifar10'):
        ds_test = tfds.load("cifar10", split='test')
        testX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        testY = np.zeros((10000,), dtype=np.int32)

        for i, elem in enumerate(ds_test):
            # print(elem)
            testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            testY[i] = elem['label']

    elif(test_ds == 'cifar100'):
        ds_test = tfds.load("cifar100", split='test')
        testX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        testY = np.zeros((10000,), dtype=np.int32)

        for i, elem in enumerate(ds_test):
            # print(elem)
            testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            testY[i] = elem['label']

    
    for i in range(n_runs):
        score = []
        # initialize model
        if(test_model == "resnet"):
            # Resnet 18 - modify stages for other architecture
            model, encoder = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate)
        elif(test_model == "wrn"):
            # Wide-Resnet 28-10 - modify for different architecture
            model, encoder = WRN(N=4, in_shape=(32,32,3), k=3, n_out=n_classes, modBlock=train_modBlock,
                                 ablate = train_ablate)
        elif(test_model == "wrn-ensemble"):
            model, encoder = ensemble_wrn(n_members, N=4, in_shape=(32,32,3), k=3, n_out=n_classes,
                                          modBlock=train_modBlock, ablate = train_ablate)
        elif(test_model == "resnet-ensemble"):
            model, encoder = ensemble_resnet(n_members, stages=[64,128],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate=train_ablate)
        else:
            print("Wrong model name chosen!")

        if(train_modBlock):
            if(train_ablate):
                ckpt_path = 'trained_models/full_models/training_'+test_model+"_"+"SN"+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else:
                ckpt_path = 'trained_models/full_models/training_'+test_model+"_"+"SN"+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"
        else:
            if(train_ablate):
                ckpt_path = 'trained_models/full_models/training_'+test_model+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else:
                ckpt_path = 'trained_models/full_models/training_'+test_model+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"
        
        if(train_modBlock):
            if(train_ablate):
                model_path = 'trained_models/full_models_afterTraining/training_'+test_model+"_"+"SN"+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else: 
                model_path = 'trained_models/full_models_afterTraining/training_'+test_model+"_"+"SN"+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"
        else:
            if(train_ablate):
                model_path = 'trained_models/full_models_afterTraining/training_'+test_model+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else: 
                model_path = 'trained_models/full_models_afterTraining/training_'+test_model+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"

        if(train_modBlock):
            if(train_ablate):
                encoder_path = 'trained_models/encoders/training_'+test_model+"_"+"SN"+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else:
                encoder_path = 'trained_models/encoders/training_'+test_model+"_"+"SN"+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"
        else:
            if(train_ablate):
                encoder_path = 'trained_models/encoders/training_'+test_model+"_"+train_ds+"_ablation"+"_n_run_"+str(i+1)+"/cp.ckpt"
            else:
                encoder_path = 'trained_models/encoders/training_'+test_model+"_"+train_ds+"_n_run_"+str(i+1)+"/cp.ckpt"
        
        model.load_weights(model_path).expect_partial()
        # model.load_weights(ckpt_path).expect_partial()
        if(test == "accuracy"):
            # load weights from i-th training run
            # checkpoints to save weights of the model
            # model.load_weights(ckpt_path).expect_partial()

            # evaluate accuracy on test_ds
            _, acc = model.evaluate(testX, testY, batch_size=batch_size)
            score.append(acc)
        
        elif(test == "ece"):
            # caluclate expected calibration error on test set
            logits = model.predict(testX, batch_size=batch_size)
            ece = tfp.stats.expected_calibration_error(num_bins = 10, logits=logits, labels_true=testY)
            score.append(ece*100)
        elif(test=="ood"):
            # run ood expirments on test set
            if(test_model == 'resnet'):
                if(uncertainty == 'softmax'):
                    test_logits = model.predict(testX, batch_size=batch_size)
                    aleatoric, epistemic = resnet_uncertainty(test_logits, mode=uncertainty)
                    print(epistemic)
                    score.append(epistemic)
                elif(uncertainty == 'energy'):
                    test_logits = model.predict(testX, batch_size=batch_size)
                    aleatoric, epistemic = resnet_uncertainty(test_logits, mode=uncertainty)
                    score.append(epistemic)
                elif(uncertainty == 'DDU'):
                    pass
            elif(test_model == 'wrn'):
                if(uncertainty == 'softmac'):
                    pass
                elif(uncertainty == 'energy'):
                    pass
                elif(uncertainty == 'DDU'):
                    pass

    print("Mean score %s:  %f" %(test,np.mean(score)))
