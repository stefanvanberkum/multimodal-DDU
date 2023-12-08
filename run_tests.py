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
from WRN import WRN, wrn_uncertainty, WRN_with_augment
from resNet import resnet, resnet_uncertainty
from ensembles import ensemble_resnet, ensemble_wrn, ensemble_uncertainty
from uncertainty import DDU, DDU_KD, DDU_CWKD, DDU_VI
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
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
parser.add_argument("--temperature_scaling", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--temperature", default = 1.0, type=float)
parser.add_argument("--temperature_criterion", default='ece', type=str)
parser.add_argument("--data_augment", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--batch_norm_momentum", default=0.99, type=float)
parser.add_argument("--dropout", default=0.0, type=float)


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
    temp_scaling_split = 0.1
    temp = args.temperature
    temp_criterion = args.temperature_criterion
    data_augment = args.data_augment
    batch_norm_momentum = args.batch_norm_momentum
    dropout = args.dropout


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
    # fp_rates = []
    # tp_rates = []
    # precisions = []
    # recalls = []
    auprc_scores = []
    all_epistemics_in = []
    all_epistemics_out = []
    all_aleatorics_in = []
    all_aleatorics_out = []

    for i in range(n_runs):
        # initialize model
        if(test_model == "resnet"):
            # Resnet 18 - modify stages for other architecture
            model, encoder = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate)
        elif(test_model == "wrn"):
            # Wide-Resnet 28-10 - modify for different architecture
            if(data_augment):
                model, encoder = WRN_with_augment(N=4, in_shape=(32,32,3), k=3, n_out=n_classes, dropout=dropout,data_augment=data_augment, modBlock=train_modBlock, ablate = train_ablate, batch_norm_momentum=batch_norm_momentum)
            else:
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
                    score.append(100*auroc)
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
                    score.append(100*auroc)
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
                    # print("Calculate features")
                    features_in = encoder.predict(testX, batch_size=batch_size)
                    features_out = encoder.predict(oodX, batch_size=batch_size)

                    aleatoric_in, epistemic_in = ddu.predict(features_in,probs_in)
                    epistemic_in[epistemic_in == np.inf] = np.finfo(np.float32).max
                    epistemic_in[epistemic_in == -np.inf] = np.finfo(np.float32).min
                    # print("Epistemic in: ", epistemic_in)
                    aleatoric_out, epistemic_out = ddu.predict(features_out, probs_out)
                    epistemic_out[epistemic_out == np.inf] = np.finfo(np.float32).max
                    epistemic_out[epistemic_out == -np.inf] = np.finfo(np.float32).min
                    # print("Epistemic out: ", epistemic_out)

                    epistemic = np.concatenate([-epistemic_in, -epistemic_out], axis=0)

                    # calculate auroc score
                    # print("Before aruoc!")
                    # compute precision and recall for error analysis
                    # fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=epistemic)
                    # precision, recall, pr_thresholds = precision_recall_curve(y_true=labels, probas_pred=epistemic) 
                    auroc = roc_auc_score(y_true = labels, y_score=epistemic)
                    print("After Auroc! Auroc is: %f" % auroc)
                    auprc = average_precision_score(y_true = labels, y_score= epistemic)
                    # print("Tpr: ", tpr)
                    # print("Fpr: ", fpr)


                    # print("Epistemic: ", epistemic)

                    # append auroc score to list
                    score.append(auroc*100)
                    auprc_scores.append(auprc*100)
                    # tp_rates.append(tpr)
                    # fp_rates.append(fpr)
                    # precisions.append(precision)
                    # recalls.append(recall)

                elif(uncertainty == 'plotDDU'):
                    labels_in = np.ones(np.shape(testY))
                    labels_out = np.zeros(np.shape(oodY)) 
                    labels = np.concatenate([labels_in, labels_out], axis=0)
                    probs_in = softmax(model.predict(testX, batch_size=batch_size)/temp, axis=-1)
                    probs_out = softmax(model.predict(oodX, batch_size=batch_size)/temp, axis=-1)
                    train_features = encoder.predict(trainX, batch_size=batch_size)
                    features_in = encoder.predict(testX, batch_size=batch_size)
                    features_out = encoder.predict(oodX, batch_size=batch_size)

                    ddu = DDU(train_features, trainY)

                    aleatoric_in, epistemic_in = ddu.predict(features_in,probs_in)
                    epistemic_in[epistemic_in == np.inf] = np.finfo(np.float32).max
                    epistemic_in[epistemic_in == -np.inf] = np.finfo(np.float32).min
                    # print("Epistemic in: ", epistemic_in)
                    aleatoric_out, epistemic_out = ddu.predict(features_out, probs_out)
                    epistemic_out[epistemic_out == np.inf] = np.finfo(np.float32).max
                    epistemic_out[epistemic_out == -np.inf] = np.finfo(np.float32).min

                    # append to lists
                    all_epistemics_in.append(epistemic_in)
                    all_epistemics_out.append(epistemic_out)
                    all_aleatorics_in.append(aleatoric_in)
                    all_aleatorics_out.append(aleatoric_out)


    print("Mean score %s:  %f" %(test,np.mean(score)))
    print("Std score %s: %f" % (test, np.std(score)))
    if(auprc_scores):
        print("Mean AUPRC-score:  %f" %(np.mean(auprc_scores)))
        print("Std AUPRC-score: %f" % (np.std(auprc_scores)))

    if(all_epistemics_in):
        np.savez('/home/jacobbrandauer/densities_and_entropies/'+test_model+"_"+"SN"+"_"+train_ds+"_vs_"+test_ds+".npz", array1=all_epistemics_in, array2=all_epistemics_out, array3=all_aleatorics_in, array4=all_aleatorics_out)
        print("Results saved to path: /home/jacobbrandauer/densities_and_entropies/"+test_model+"_"+"SN"+"_"+train_ds+"_vs_"+test_ds+".npz")

    

    # if(tp_rates): 
    #     print("Mean TP-rate:  %f" %(np.mean(tp_rates)))
    #     print("Std TP-rate: %f" % (np.std(tp_rates)))
    # if(fp_rates): 
    #     print("Mean FP-rate:  %f" %(np.mean(fp_rates)))
    #     print("Std FP-rate: %f" % (np.std(fp_rates)))
    # if(precisions):
    #     print("Mean precisione:  %f" %(np.mean(precisions)))
    #     print("Std precisions: %f" % (np.std(precisions)))
    # if(recalls):
    #     print("Mean recall:  %f" %(np.mean(recalls)))
    #     print("Std recall: %f" % (np.std(recalls)))
    
