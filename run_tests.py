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
from scipy.special import softmax
import datasets

# parameters for testing
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str) # 'wrn', 'resnet', 'wrn-ensemble', 'resnet-ensemble'
parser.add_argument("--train_ds", required=True, type=str) # 'cifar10', 'cifar100'
parser.add_argument("--test_ds", required=True, type=str) # 'cifar100', 'SVHN', 'imageNet'
parser.add_argument("--modBlock", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ablate", default=False, action=argparse.BooleanOptionalAction)
# parser.add_argument("--n_epochs", default=350, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--test", default = "accuracy", type=str) # 'accuracy', 'ece', 'ood'
parser.add_argument("--n_runs", default = 5, type=int) # number of training runs to average over
parser.add_argument("--uncertainty", default='DDU', type=str) # 'energy', 'softmax', 'DDU', 'KD', 'CWKD', 'VI'
parser.add_argument("--temperature_scaling", default=True, type=bool)
parser.add_argument("--temperature", default = 1.0, type=float)
parser.add_argument("--temperature_criterion", default='ece', type=str)


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
    temperature_scaling = args.temperature_scaling
    temp_scaling_split = 0.2
    temp = args.temperature
    temp_criterion = args.temperature_criterion

    if(train_ds == 'cifar10'):
        n_classes = 10
        ds_train = tfds.load("cifar10", split='train')
        trainX = np.zeros((50000, 32, 32,3), dtype=np.float32)
        trainY = np.zeros((50000,), dtype=np.int32)
        ds_test = tfds.load("cifar10", split='test')
        testX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        testY = np.zeros((10000,), dtype=np.int32)

        # sample 10-percent for temperate scaling
        valX = np.zeros((int(temp_scaling_split*50000),32,32,3), dtype=np.float32)
        valY = np.zeros((int(temp_scaling_split*50000),), dtype=np.int32)
        valIndices = np.random.choice(50000, int(temp_scaling_split*50000), replace=False)
        valCount = 0
        for i, elem in enumerate(ds_train):
            # print(elem)
            trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            trainY[i] = elem['label']
            # if(np.isin(valIndices, i).all()):
            if(i in valIndices):
                # print("IN !")
                valX[valCount, :, : ,: ] = tf.cast(elem['image'], tf.float32)/255.
                valY[valCount] = elem['label']
                valCount += 1
        for i, elem in enumerate(ds_test):
            testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            testY[i] = elem['label']

    elif(train_ds == 'cifar100'):
        n_classes = 100
        ds_train = tfds.load('cifar100', split='train')
        trainX = np.zeros((50000, 32, 32,3), dtype=np.float32)
        trainY = np.zeros((50000,), dtype=np.int32)
        ds_test = tfds.load("cifar100", split='test')
        testX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        testY = np.zeros((10000,), dtype=np.int32)
        for i, elem in enumerate(ds_train):
            trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            trainY[i] = elem['label']
        for i, elem in enumerate(ds_test):
            testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            testY[i] = elem['label']

    if(test_ds == 'cifar10'):
        ds_ood = tfds.load("cifar10", split='test')
        oodX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        oodY = np.zeros((10000,), dtype=np.int32)
        for i, elem in enumerate(ds_ood):
            # print(elem)
            oodX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            oodY[i] = elem['label']

    elif(test_ds == 'cifar100'):
        ds_ood = tfds.load("cifar100", split='test')
        oodX = np.zeros((10000, 32, 32,3), dtype=np.float32)
        oodY = np.zeros((10000,), dtype=np.int32)

        for i, elem in enumerate(ds_ood):
            # print(elem)
            oodX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
            oodY[i] = elem['label']
    elif(test_ds == 'svhn'):
        ds_ood = tfds.load('svhn_cropped', split='test')
        oodX = np.zeros((26032, 32, 32, 3), dtype=np.float32)
        oodY = np.zeros((26032,), dtype=np.int32)
        for i, elem in enumerate(ds_ood):
            oodX[i,:,:,:] = tf.cast(elem['image'], tf.float32)/255.
            oodY[i] = elem['label']
    
    elif(test_ds == 'imageNet'):
        
        # load tiny-image-net dataset from huggingface
        ds_ood= datasets.load_dataset('Maysee/tiny-imagenet', split='valid')
        # ds = datasets.Dataset.from_dict(data)
        ds_ood_tf = ds_ood.with_format("tf")
        # print("-----Dataset-----")
        # print(ds_ood_tf[0]['image'])
        # print("------------")
        oodX = np.zeros((10000, 32, 32, 3), dtype=np.float32)
        oodY = np.zeros((10000,), dtype=np.int32)
        wrongShapeIndices = []
        count = 0
        for i, elem in enumerate(ds_ood_tf):
            image = tf.cast(elem['image'], tf.float32)/255.

            # check if image has only one channel
            if(tf.shape(image).shape[0] == 2):
                # print("Shape: ",tf.shape(image).shape)
                reshaped_image = tf.image.resize(tf.expand_dims(image, axis=-1), [32,32]).numpy()

                # repeat grayscale images along all three channels
                oodX[i, :, :, 0] = reshaped_image[:,:,0]
                oodX[i, :, :, 1] = reshaped_image[:,:,0]
                oodX[i, :, :, 2] = reshaped_image[:,:,0]
                oodY[i] = elem['label']
                wrongShapeIndices.append(i)
                continue
            image = tf.cast(elem['image'], tf.float32)/255.
            oodX[i,:,:,:] = tf.image.resize(tf.cast(elem['image'], tf.float32)/255., [32,32]).numpy()
            oodY[i] = elem['label']
            count += 1

    score = []
    temps = []
    for i in range(n_runs):
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
        # model.load_weights('trained_models/full_models_afterTraining/training_resnet_SN_cifar10_n_run_1/cp.ckpt').expect_partial()

        if(temperature_scaling):
            # perform temperature scaling to calibrate network
            temp_step = 0.1
            num_temps = 100
            T = 0.1
            opt_temp = T
            # opt_ece = 1e+03
            opt_nll = 1e7
            opt_ece = 1e7

            for i in range(num_temps):
                temp_logits = model.predict(valX, batch_size=batch_size)/T
                if(temp_criterion == 'ece'):
                    ece_temp = 100*tfp.stats.expected_calibration_error(num_bins = 10, logits=temp_logits, labels_true=valY)
                    if(ece_temp < opt_ece):
                        opt_ece = ece_temp
                        opt_temp = T
                elif(temp_criterion == 'nll'):
                    nll_temp = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valY, logits=temp_logits)).numpy()
                    # print("NLL: ", nll_temp)
                    if(nll_temp < opt_nll):
                        opt_nll = nll_temp
                        opt_temp = T
                else: 
                    nll_temp = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valY, logits=temp_logits)).numpy()
                    # print("NLL: ", nll_temp)
                    if(nll_temp < opt_nll):
                        opt_nll = nll_temp
                        opt_temp = T
                # print("Temp: ", T)
                T += temp_step
            temp = opt_temp
            temps.append(temp)
            print("Optimal Temperature: %f with optimal NLL: %f"%(temp, opt_nll))
            

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
            ece = tfp.stats.expected_calibration_error(num_bins = 10, logits=logits/temp, labels_true=testY)
            score.append(ece*100)
        elif(test=="ood"):
            # logits_in = model.predict(trainX, batch_size=batch_size)
            # logits_out = model.predict(testX, batch_size=batch_size)
            # logits = np.concatenate([logits_in, logits_out], axis=0)
            # run ood expirments on test set
            if(test_model == 'resnet'):
                if(uncertainty == 'softmax'):
                    labels_in = np.zeros(np.shape(testY))
                    labels_out = np.ones(np.shape(oodY))
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    logits_in = model.predict(testX, batch_size=batch_size) 
                    logits_out = model.predict(oodX, batch_size=batch_size) # TODO: change to ood data
                    aleatoric_out, epistemic_out = resnet_uncertainty(logits_out/temp, mode=uncertainty)
                    aleatoric_in, epistemic_in = resnet_uncertainty(logits_in/temp, mode=uncertainty)
                    epistemic = np.concatenate([epistemic_in, epistemic_out], axis=0)
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic)
                    # print(epistemic)
                    score.append(auroc*100)
                elif(uncertainty == 'energy'):
                    # define labels for in-distribution and out-of-distribution data
                    labels_in = np.zeros(np.shape(testY))
                    labels_out = np.ones(np.shape(oodY))
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    logits_in = model.predict(testX, batch_size=batch_size)
                    logits_out = model.predict(oodX, batch_size=batch_size) 
                    aleatoric_out, epistemic_out = resnet_uncertainty(logits_out/temp, mode=uncertainty)
                    aleatoric_in, epistemic_in = resnet_uncertainty(logits_in/temp, mode=uncertainty)
                    epistemic = np.concatenate([epistemic_in, epistemic_out], axis=0)
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic)
                    score.append(auroc*100)
                elif uncertainty == 'DDU' or uncertainty == 'KD' or uncertainty == 'CWKD' or uncertainty == 'VI':
                    # define labels for in-distribution and out-of-distribution data
                    labels_in = np.ones(np.shape(testY)) # define in-distribution data as ones - DDU estimates probability of being in-distribution data
                    labels_out = np.zeros(np.shape(oodY)) 
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    probs_in = softmax(model.predict(testX, batch_size=batch_size)/temp, axis=-1)
                    probs_out = softmax(model.predict(oodX, batch_size=batch_size)/temp, axis=-1) 
                    # map training samples to feature space to fit estimator
                    train_features = encoder.predict(trainX, batch_size=batch_size) 
                    # print("Test y: ", np.unique(testY))

                    if uncertainty == 'DDU':
                        ddu = DDU(train_features, trainY)
                    elif uncertainty == 'KD':
                        ddu = DDU_KD(train_features)
                    elif uncertainty == 'CWKD':
                        ddu = DDU_CWKD(train_features, trainY)
                    elif uncertainty == 'VI':
                        ddu = DDU_VI(train_features, 10 * n_classes)
                    
                    # predict uncertainty on in-distribution and out-of-distribution data
                    features_in = encoder.predict(testX)
                    featoures_out = encoder.predict(oodX)
                    aleatoric_in, epistemic_in = ddu.predict(features_in,probs_in)
                    aleatoric_out, epistemic_out = ddu.predict(featoures_out, probs_out)
                    epistemic = np.concatenate([-epistemic_in, -epistemic_out], axis=0)

                    # calculate auroc score
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic) 

                    # print("Epistemic: ", epistemic)

                    # append auroc score to list
                    score.append(auroc*100)

            elif(test_model == 'wrn'):
                if(uncertainty == 'softmax'):
                    labels_in = np.zeros(np.shape(testY))
                    labels_out = np.ones(np.shape(oodY))
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    logits_in = model.predict(testX, batch_size=batch_size) 
                    logits_out = model.predict(oodX, batch_size=batch_size) 
                    aleatoric_out, epistemic_out = wrn_uncertainty(logits_out/temp, mode=uncertainty)
                    aleatoric_in, epistemic_in = wrn_uncertainty(logits_in/temp, mode=uncertainty)
                    epistemic = np.concatenate([epistemic_in, epistemic_out], axis=0)
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic)
                elif(uncertainty == 'energy'):
                    labels_in = np.zeros(np.shape(testY))
                    labels_out = np.ones(np.shape(oodY))
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    logits_in = model.predict(testX, batch_size=batch_size)
                    logits_out = model.predict(oodX, batch_size=batch_size)
                    aleatoric_out, epistemic_out = wrn_uncertainty(logits_out/temp, mode=uncertainty)
                    aleatoric_in, epistemic_in = wrn_uncertainty(logits_in/temp, mode=uncertainty)
                    epistemic = np.concatenate([epistemic_in, epistemic_out], axis=0)
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic)
                elif uncertainty == 'DDU' or uncertainty == 'KD' or uncertainty == 'CWKD' or uncertainty == 'VI':
                    # define labels for in-distribution and out-of-distribution data
                    labels_in = np.ones(np.shape(testY))
                    labels_out = np.zeros(np.shape(oodY)) 
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    probs_in = softmax(model.predict(testX, batch_size=batch_size)/temp, axis=-1)
                    probs_out = softmax(model.predict(oodX, batch_size=batch_size)/temp, axis=-1)

                    # map training samples to feature space to fit estimator

                    train_features = encoder.predict(trainX, batch_size=batch_size)
                    # print("Test y: ", np.unique(testY))

                    if uncertainty == 'DDU':
                        ddu = DDU(train_features, trainY)
                    elif uncertainty == 'KD':
                        ddu = DDU_KD(train_features)
                    elif uncertainty == 'CWKD':
                        ddu = DDU_CWKD(train_features, trainY)
                    elif uncertainty == 'VI':
                        ddu = DDU_VI(train_features, 10 * n_classes)

                    # predict uncertainty on in-distribution and out-of-distribution data
                    features_in = encoder.predict(testX)
                    featoures_out = encoder.predict(oodX)
                    aleatoric_in, epistemic_in = ddu.predict(features_in,probs_in)
                    aleatoric_out, epistemic_out = ddu.predict(featoures_out, probs_out)
                    epistemic = np.concatenate([-epistemic_in, -epistemic_out], axis=0)

                    # calculate auroc score
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic) 

                    # print("Epistemic: ", epistemic)

                    # append auroc score to list
                    score.append(auroc*100)

    print("Mean score %s:  %f" %(test,np.mean(score)))
    print("Std score %s: %f" % (test, np.std(score)))

