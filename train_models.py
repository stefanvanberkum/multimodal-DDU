"""
    Script for training the models in "Deep Deterministic Uncertainty: A New Simple Baseline", Mukhoti et al. (2023)
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from WRN import WRN, wrn_uncertainty
from resNet import resnet, resnet_uncertainty
from ensembles import ensemble_resnet, ensemble_uncertainty, ensemble_wrn
import argparse


# parameters for training
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str) # 'wrn', 'resnet', 'wrn-ensemble', 'resnet-ensemble'
parser.add_argument("--dataset", required=True, type=str) # 'cifar10', 'cifar100'
parser.add_argument("--modBlock", default=True, type=bool)
parser.add_argument("--ablate", default=False, type=bool)
parser.add_argument("--n_epochs", default=350, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--n_run",required=True, type=int)


# # parameters for training
# train_model = "resnet"  # 'wrn', 'resnet', 'wrn-ensemble', 'resnet-ensemble' 
# dataset = "cifar10" # 'cifar10', 'cifar100'
# train_modBlock = True
# train_ablate = False
# n_members = 5
# batch_size = 128
# n_epochs = 1


if(__name__=="__main__"):
    args = parser.parse_args()
    train_model = args.model 
    dataset = args.dataset # 'cifar10', 'cifar100'
    train_modBlock = args.modBlock
    train_ablate = args.ablate
    n_members = 5
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_run = args.n_run

    # load training data
    if(dataset == 'cifar10'):
        ds_train = tfds.load("cifar10", split='train')

        # batch training data
        # train_batches = ds_train.shuffle(100).batch(batch_size)

        ds_test = tfds.load("cifar10", split='test')

        # test_batches = ds_test.shuffle(100).batch(batch_size)

        n_classes = 10

    elif(dataset == 'cifar100'):
        ds_train = tfds.load("cifar100", split='train')


        # batch training data
        # train_batches = ds_train.shuffle(100).batch(batch_size)

        ds_test = tfds.load("cifar100", split='test')

        # test_batches = ds_test.shuffle(100).batch(batch_size)

        n_classes = 100

    trainX = np.zeros((50000, 32,32,3), dtype=np.float32)
    trainY = np.zeros((50000,), dtype=np.int32)
    testX = np.zeros((10000, 32, 32,3), dtype=np.float32)
    testY = np.zeros((10000,), dtype=np.int32)

    for i, elem in enumerate(ds_train):
         # print(elem)
        trainX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
        trainY[i] = elem['label']
    for i, elem in enumerate(ds_test):
        # print(elem)
        testX[i, :, :, :] = tf.cast(elem['image'], tf.float32)/255.
        testY[i] = elem['label']
    # initialize model
    if(train_model == "resnet"):
        # Resnet 18 - modify stages for other architecture
        model, encoder = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate)
    elif(train_model == "wrn"):
        # Wide-Resnet 28-10 - modify for different architecture
        model, encoder = WRN(N=4, in_shape=(32,32,3), k=10, n_out=n_classes, modBlock=train_modBlock, ablate = train_ablate) 
    elif(train_model == "wrn-ensemble"):
        model = ensemble_wrn(n_members, N=4, in_shape=(32,32,3), k=10, n_out=n_classes, modBlock=train_modBlock, ablate = train_ablate)
    elif(train_model == "resnet-ensemble"):
        model = ensemble_resnet(n_members, stages=[64,128],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate=train_ablate)


    # train model

    # learning rate scheduler
    def scheduler(epoch, lr):
        if(epoch ==149 or epoch == 249):
            return lr/10
        else:
            return lr
  
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


    # checkpoints to save weights of the model
    # if(train_modBlock):
    #     if(train_ablate):
    #         ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
    #     else: 
    #         ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"
    # else:
    #     if(train_ablate):
    #         ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
    #     else: 
    #         ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"

    
    # ckpt_callback =  tf.keras.callbacks.ModelCheckpoint(
    #     filepath=ckpt_path,
    #     save_weights_only=True,
    #     save_freq='epoch')
    
    if((train_model == 'resnet') or (train_model == 'wrn')):    
        # checkpoints to save weights of the model
        if(train_modBlock):
            if(train_ablate):
                ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
            else: 
                ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"
        else:
            if(train_ablate):
                ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
            else: 
                ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"

    
        ckpt_callback =  tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_path,
                    save_weights_only=True,
                    save_freq='epoch')

        model.fit(x=trainX, y=trainY, epochs=n_epochs, batch_size = batch_size, callbacks=[lr_callback, ckpt_callback], shuffle=True)
        if(train_modBlock):
            if(train_ablate):
                model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
        else:
            if(train_ablate):
                model_path = 'trained_models/full_models_afterTraing/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
        model.save_weights(model_path)

        # save encoder in different files
        if(train_modBlock):
            if(train_ablate):
                encoder_path = 'trained_models/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                encoder_path = 'trained_models/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
        else:
            if(train_ablate):
                encoder_path = 'trained_models/encoders/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                encoder_path = 'trained_models/encoders/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"

        # save encoder weights
        encoder.save_weights(encoder_path)
    elif((train_model == 'wrn-ensemble') or (train_model == 'resnet-ensemble')):
        for j in range(n_members):
            # checkpoints to save weights of the model
            if(train_modBlock):
                if(train_ablate):
                    ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
                else: 
                    ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
            else:
                if(train_ablate):
                    ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
                else: 
                    ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"

            ckpt_callback =  tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                save_freq='epoch')

            member = model[j]
            # encoder_member = encoder[j]

            member.fit(x=trainX, y=trainY, epochs=n_epochs, batch_size = batch_size, callbacks=[lr_callback, ckpt_callback], shuffle=True)
            if(train_modBlock):
                if(train_ablate):
                    model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
                else: 
                    model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
            else:
                if(train_ablate):
                    model_path = 'trained_models/full_models_afterTraing/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
                else: 
                    model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
            
            member.save_weights(model_path)

            # # save encoder in different files
            # if(train_modBlock):
            #     if(train_ablate):
            #         ncoder_path = 'trained_models/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
            #     else: 
            #         encoder_path = 'trained_models/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
            # else:
            #     if(train_ablate):
            #         encoder_path = 'trained_models/encoders/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
            #     else: 
            #         encoder_path = 'trained_models/encoders/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"

            # # save encoder weights
            # encoder_member.save_weights(encoder_path)
    else: 
        print("ERROR! Wrong model choice!")




    # # # load weights
    # model2, encoder2 = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate)
    # model2.load_weights(model_path).expect_partial()
    # model2.evaluate(x = testX, y=testY, batch_size=batch_size)


    # features1=encoder2(tf.expand_dims(trainX[0], axis=0))
    # print("Features 1: ", features1)
    # encoder2.load_weights(encoder_path)
    # features2=encoder2(tf.expand_dims(trainX[0], axis=0))
    # print("Features 2: ", features2)

    # # check if features are equal
    # if((features1.numpy() == features2.numpy()).all()):
    #     print("Same!!")
    # else: 
    #     print("Not the same!!")
    # initialize model
    # if(train_model == "resnet"):
    #     # Resnet 18 - modify stages for other architecture
    #     model, encoder = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate)
    # elif(train_model == "wrn"):
    #     # Wide-Resnet 28-10 - modify for different architecture
    #     model, encoder = WRN(N=4, in_shape=(32,32,3), k=10, n_out=n_classes, modBlock=train_modBlock, ablate = train_ablate) 
    # elif(train_model == "wrn-ensemble"):
    #     model = ensemble_wrn(n_members, N=4, in_shape=(32,32,3), k=10, n_out=n_classes, modBlock=train_modBlock, ablate = train_ablate)
    # elif(train_model == "resnet-ensemble"):
    #     model = ensemble_resnet(n_members, stages=[64,128],N=2,in_filters=64, in_shape=(32,32,3), n_out = n_classes, modBlock = train_modBlock, ablate=train_ablate)

    # # load ensemble weights
    # for j in range(n_members):
    #     member = model[j]
    #     if(train_modBlock):
    #         if(train_ablate):
    #             ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
    #         else: 
    #             ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
    #     else:
    #         if(train_ablate):
    #             ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
    #         else: 
    #             ckpt_path = 'trained_models/full_models/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/cp.ckpt"
    #     if(train_modBlock):
    #         if(train_ablate):
    #             model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
    #         else: 
    #             model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
    #     else:
    #         if(train_ablate):
    #             model_path = 'trained_models/full_models_afterTraing/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
    #         else: 
    #             model_path = 'trained_models/full_models_afterTraining/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"_member_"+str(j+1)+"/checkpoint"
    #     print("Model path: ", model_path)
    #     member.load_weights(model_path).expect_partial()
    #     member.evaluate(x = testX, y=testY)

