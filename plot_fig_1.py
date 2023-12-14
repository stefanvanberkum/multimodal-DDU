"""
Reproduction of Figure 1 from "Deep Deterministic Uncertainty: A New Simple Baseline",               et al. (2023)
"""

#Import libraries
import tensorflow as tf
import tensorflow_probability as tfp
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from resNet import resnet
from VGG16 import vgg_16
from LeNet import lenet
import argparse
import ddu_dirty_mnist
import uncertainty
from scipy.special import softmax
import ssl
import pandas as pd

#Fix SSL Error
ssl._create_default_https_context = ssl._create_unverified_context  

#Parameters for training
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet", type=str) # 'resnet', 'vgg_16', 'lenet'
parser.add_argument("--n_epochs", default=50, type=int)
parser.add_argument("--mode", default="plot", type=str) # "train", "calculate", "calculate_and_plot", "all"
parser.add_argument("--temperature_scaling", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--n_run", default=1, type=int)
parser.add_argument("--modBlock", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ablate", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--temp_criterion", default="ece", type=str) # "ece", "nll"

dataset = "dirty_mnist" 
batch_size = 128
n_classes = 10

if(__name__=="__main__"):
    args = parser.parse_args()
    train_model = args.model 
    n_epochs = args.n_epochs
    mode = args.mode
    temperature_scaling = args.temperature_scaling
    n_run = args.n_run
    train_modBlock = args.modBlock
    train_ablate = args.ablate
    temp_criterion = args.temp_criterion
    
    def convert_torch_to_tensorflow_dataset(concatenated_dataset):
        """
        Converts concatenated torch dataset to Tensorflow dataset (Relevant for Ambiguous MNIST and Dirty MNIST)
        Returns tensorflow dataset
        """

        data = []
        labels = []
        
        for sample in concatenated_dataset:
            data_batch, label_batch = sample
            data.append(data_batch.numpy())
            labels.append(label_batch.numpy())
        
        # Convert to TensorFlow tensors
        data_tf = tf.convert_to_tensor(np.array(data), dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(np.array(labels), dtype=tf.int32)
        
        # Create a TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((data_tf, labels_tf))
        
        return dataset

    def create_arrays(datset, temperature_scaling = False):
        """
        Input train/test dataset
        Set Temperature Scaling,  Default = False
        Returns numpy test and val X and Y
        """

        #Create tensorflow train/test dataset
        datset = convert_torch_to_tensorflow_dataset(datset)
        
        if not temperature_scaling:
            dataX = np.zeros((len(datset), 28,28), dtype=np.float64)
            dataY = np.zeros((len(datset),), dtype=np.int32)

            for i, elem in enumerate(datset):
                dataX[i, :, :] = tf.cast(elem[0], tf.float32) / 255.
                dataY[i] = elem[1]

            return dataX, dataY

        else: #Temperature Scaling
            temp_scaling_split = 0.1
            seed = 10

            array_length = len(datset)
            train_length = int(array_length*(1-temp_scaling_split))
            val_length = int(array_length-train_length)

            dataX = np.zeros((train_length, 28,28), dtype=np.float64)
            dataY = np.zeros((train_length,), dtype=np.int32)

            valX = np.zeros((val_length, 28,28), dtype=np.float64)
            valY = np.zeros((val_length,), dtype=np.int32)

            #Set seed for reproducability
            np.random.seed(seed=seed)
            valIndices = np.random.choice(array_length, val_length, replace=False)
            valCount = 0
            trainCount = 0
            for i, elem in enumerate(datset):
                if(i in valIndices):
                    valX[valCount, :, :] = tf.cast(elem[0], tf.float32) / 255.
                    valY[valCount] = elem[1]
                    valCount += 1
                else:
                    dataX[trainCount, :, :] = tf.cast(elem[0], tf.float32) / 255.
                    dataY[trainCount] = elem[1]
                    trainCount += 1
            
            return dataX, dataY, valX, valY
    
    def train_model_function(trainX, trainY):
        """
        Train model and save the weights resnet, vgg_16, lenet
        Returns model_path, encoder_path
        """
        
        # initialize models
        if(train_model == "resnet"):
            # Resnet 18 - modify stages for other architecture
            model, encoder = resnet(stages = [64,128,256,512], N = 2, in_filters = 64, in_shape = (28,28,1), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate, coeff = 3)
        elif(train_model == "vgg_16"):
            #VGG 16
            model, encoder = vgg_16(in_shape = (28,28,1), stages=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], kernel_size = (3,3), padding = "same", modBlock = train_modBlock, ablate = train_ablate, n_out = n_classes)
        elif(train_model == "lenet"):
            #LeNet
            model, encoder = lenet(in_shape = (28,28,1), kernel_size = (5,5), padding = "same", activation = "relu", modBlock = train_modBlock, ablate = train_ablate, n_out = n_classes)

        # checkpoints to save weights of the model
        if(train_modBlock):
            if(train_ablate):
                ckpt_path = 'results/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
            else: 
                ckpt_path = 'results/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"
        else:
            if(train_ablate):
                ckpt_path = 'results/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/cp.ckpt"
            else: 
                ckpt_path = 'results/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/cp.ckpt"
        
        ckpt_callback =  tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            save_freq='epoch')
        
        # learning rate scheduler
        def scheduler(epoch, lr):
            if(epoch ==24 or epoch == 39):
                return lr/10
            else:
                return lr
  
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

        #Train model
        model.fit(x=trainX, y=trainY, epochs=n_epochs, batch_size = batch_size, callbacks=[lr_callback, ckpt_callback], shuffle=True)
        if(train_modBlock):
            if(train_ablate):
                model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
        else:
            if(train_ablate):
                model_path = 'results/full_models_afterTraing/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"

        #Save model weights
        model.save_weights(model_path)

        #Save encoder in different files
        if(train_modBlock):
            if(train_ablate):
                encoder_path = 'results/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                encoder_path = 'results/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
        else:
            if(train_ablate):
                encoder_path = 'results/encoders/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
            else: 
                encoder_path = 'results/encoders/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"

        #Save encoder weights
        encoder.save_weights(encoder_path)

        print("Model saved to path", model_path)
        print("Encoder saved to path", encoder_path)

        return model_path, encoder_path
    
    def load_model_and_encoder_weights(model_path, encoder_path):
        """
        Load model and encoder weights resnet, vgg_16, lenet
        Returns model and encoder
        """
        #Load weights 
        if(train_model == "resnet"):
            # Resnet 18 - modify stages for other architecture
            model, encoder = resnet(stages=[64,128,256,512],N=2,in_filters=64, in_shape=(28,28,1), n_out = n_classes, modBlock = train_modBlock, ablate = train_ablate, coeff = 3)
        elif(train_model == "vgg_16"):
            #VGG 16
            model, encoder = vgg_16(in_shape = (28,28,1), stages=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], kernel_size = (3,3), padding = "same", modBlock = train_modBlock, ablate = train_ablate, n_out = n_classes)
        elif(train_model == "lenet"):
            #LeNet
            model, encoder = lenet(in_shape = (28,28,1), kernel_size = (5,5), padding = "same", activation = "relu", modBlock = train_modBlock, ablate = train_ablate, n_out = n_classes)
        
        model.load_weights(model_path).expect_partial()
        encoder.load_weights(encoder_path).expect_partial()

        return model, encoder

    def load_datasets(dataset):
        """
        Load mnist, fashion_mnist and ambiguous_mnist datsets for evaluation
        Return testX
        """

        if dataset == "mnist":

            mnist_test = datasets.MNIST(root="~/datasets/mnist",train=False,download=True,transform=ToTensor())

            #Initialize train/test x/y numpy arrays
            testX = np.zeros((len(mnist_test), 28,28), dtype=np.float64)

            for i, elem in enumerate(mnist_test):
                testX[i, :, :] = tf.cast(elem[0], tf.float32) /255.

        if dataset == "fashion_mnist":
            fashion_mnist_test = datasets.FashionMNIST(root="~/datasets/fashion_mnist",train=False,download=True,transform=ToTensor())

            #Initialize train/test x/y numpy arrays
            testX = np.zeros((len(fashion_mnist_test), 28,28), dtype=np.float64) 

            for i, elem in enumerate(fashion_mnist_test):
                testX[i, :, :] = tf.cast(elem[0], tf.float32) / 255.

        if dataset == "ambiguous_mnist":
            # Load the Ambiguous MNIST dataset
            ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST("~/datasets/ambiguous_mnist", train=False, download=True)
            testX, _ = create_arrays(ambiguous_mnist_test)

        return testX  

    def plot_entropy(results_1, results_2, results_3):
        """
        Plots entropy of LeNet, VGG16, and ResNet18
        """
        
        bin_num = 15
        stat = "probability"
        alpha = 0.8
        binrange = [0, 2.5]

        #Create figure with 500 dpi
        sb.set()
        plt.figure(dpi=500)
            
        train_models = [results_1, results_2, results_3]
        model_names = ["LeNet", "VGG-16", "ResNet-18+SN"]

        #create subplots for LeNet, VGG16, and ResNet18
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6))

        for idx, ax in enumerate([ax1, ax2, ax3]):
            ax.grid(color='lightgray') #Set grid color
            ax.set_facecolor('white') #Set background color
            
            #Setting up data and categories for Dirty Mnist and Ambiguous Mnist
            train_model_result = train_models[idx]
            categories = np.concatenate((np.zeros_like(train_model_result.entropy_mnist, dtype=np.int8), np.ones_like(train_model_result.entropy_ambiguous_mnist, dtype=np.int8)))
            combined_data = np.concatenate((train_model_result.entropy_mnist, train_model_result.entropy_ambiguous_mnist))
            data_df = pd.DataFrame({
                                    "entropy":combined_data, 
                                    "category": categories
                                    })
            #Fill
            sb.histplot(ax = ax, data = train_model_result.entropy_fashion_mnist, stat=stat, bins = bin_num, label = "Fashion Mnist", alpha = alpha, element = "step", binrange = binrange, multiple = "stack", edgecolor = "darkorange", color = "darkorange")
            sb.histplot(ax = ax, data = data_df, x = "entropy", hue = "category" ,stat="probability", bins = bin_num, label = ["Dirty Mnist", "Ambiguous Mnist"], element = "step", binrange = binrange, multiple = "stack", palette = ["dodgerblue", "mediumblue"]) #['#3B75AF', '#1E77B4']
            
            #Lines
            sb.histplot(ax = ax, data = data_df, x = "entropy", hue = "category" ,stat="probability", bins = bin_num, legend = False, element = "step", binrange = binrange, multiple = "stack", palette = ["dodgerblue", "mediumblue"], linewidth = 3, fill = False) #['#3B75AF', '#1E77B4']
            sb.histplot(ax = ax, data = train_model_result.entropy_fashion_mnist, stat=stat, bins = bin_num, alpha = alpha, element = "step", binrange = binrange, multiple = "stack", color = "darkorange", legend= False, linewidth = 3, fill = False)

            ax.set_xlabel("Entropy " + model_names[idx])
            ax.set_ylabel("Fraction")
        
        #Legend - Loc
        legend_1 = ax1.legend(loc = "upper left")
        legend_2 = ax2.legend(loc = "upper right")
        legend_3 = ax3.legend(loc = "upper right")

        #Legend - Set Egdecolor for Markers
        for i, handle in enumerate(legend_1.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        for i, handle in enumerate(legend_2.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        for i, handle in enumerate(legend_3.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        #Legend - Label
        legend_1.get_texts()[2].set_text('Dirty Mnist')
        legend_1.get_texts()[1].set_text('Ambiguous Mnist')
        legend_2.get_texts()[2].set_text('Dirty Mnist')
        legend_2.get_texts()[1].set_text('Ambiguous Mnist')
        legend_3.get_texts()[2].set_text('Dirty Mnist')
        legend_3.get_texts()[1].set_text('Ambiguous Mnist')

        #save figure and show
        plt.savefig("entropy_all_models.png")
        plt.show()

        return


    
    def plot_density(results_1, results_2, results_3):
        """
        Plot density of LeNet, VGG16, and ResNet18
        """

        bin_num = 30
        stat = "probability"
        alpha = 0.8
        binrange = [[-250, 100], [1000, 3000], [-1000, 2500]]

        #Create figure with 500 dpi
        sb.set()
        plt.figure(dpi=500)
            
        train_models = [results_1, results_2, results_3]
        model_names = ["LeNet", "VGG-16", "ResNet-18+SN"]

        #create subplots for LeNet, VGG16, and ResNet18
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6))

        for idx, ax in enumerate([ax1, ax2, ax3]):
            ax.grid(color='lightgray') #Set grid color
            ax.set_facecolor('white') #Set background color

            #Setting up data and categories for Dirty Mnist and Ambiguous Mnist
            train_model_result = train_models[idx]
            categories = np.concatenate((np.zeros_like(train_model_result.log_gmm_density_mnist, dtype=np.int8), np.ones_like(train_model_result.log_gmm_density_ambiguous_mnist, dtype=np.int8)))
            combined_data = np.concatenate((train_model_result.log_gmm_density_mnist, train_model_result.log_gmm_density_ambiguous_mnist))
            data_df = pd.DataFrame({
                                    "density":combined_data, 
                                    "category": categories
                                    })

            #Fill
            sb.histplot(ax = ax, data = train_model_result.log_gmm_density_fashion_mnist, stat=stat, bins = bin_num, label = "Fashion Mnist", alpha = alpha, element = "step", binrange = binrange[idx], multiple = "stack", edgecolor = "darkorange", color = "darkorange")
            sb.histplot(ax = ax, data = data_df, x = "density", hue = "category" ,stat="probability", bins = bin_num, label = ["Dirty Mnist", "Ambiguous Mnist"], element = "step", binrange = binrange[idx], multiple = "stack", palette = ["dodgerblue", "mediumblue"]) 
            
            #Lines
            sb.histplot(ax = ax, data = data_df, x = "density", hue = "category" ,stat="probability", bins = bin_num, legend = False, element = "step", binrange = binrange[idx], multiple = "stack", palette = ["dodgerblue", "mediumblue"], linewidth = 3, fill = False) 
            sb.histplot(ax = ax, data = train_model_result.log_gmm_density_fashion_mnist, stat=stat, bins = bin_num, alpha = alpha, element = "step", binrange = binrange[idx], multiple = "stack", color = "darkorange", legend= False, linewidth = 3, fill = False)

            ax.set_xlabel("Density " + model_names[idx])
            ax.set_ylabel("Fraction")

        #Legend - Loc
        legend_1 = ax1.legend(loc = "upper left")
        legend_2 = ax2.legend(loc = "upper left")
        legend_3 = ax3.legend(loc = "upper left")
        
        #Legend - Set Egdecolor for Markers
        for i, handle in enumerate(legend_1.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        for i, handle in enumerate(legend_2.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        for i, handle in enumerate(legend_3.legendHandles):
            if i ==1:
                handle.set_edgecolor('mediumblue')
            if i ==2:
                handle.set_edgecolor('dodgerblue')
            else: continue

        legend_1.get_texts()[2].set_text('Dirty Mnist')
        legend_1.get_texts()[1].set_text('Ambiguous Mnist')
        legend_2.get_texts()[2].set_text('Dirty Mnist')
        legend_2.get_texts()[1].set_text('Ambiguous Mnist')
        legend_3.get_texts()[2].set_text('Dirty Mnist')
        legend_3.get_texts()[1].set_text('Ambiguous Mnist')

        #save figure
        plt.savefig("density_all_models.png")
        plt.show()

        return

    def get_entropy_log_gmm_density(datasetX, model, encoder, ddu_train_dataset, temperature = 1):
        """
        Input: datasetX, model, encoder, ddu_train_dataset, temperature
        Calculates the entropy and log_gmm_density
        """

        #Get features and y
        logits = model.predict(datasetX) / temperature
        features = encoder.predict(datasetX)
        y = softmax(logits, axis=-1)

        entropy, gmm_density = ddu_train_dataset.predict(features, y) 

        return entropy, gmm_density

    def get_all_entropy_log_gmm_density(model, encoder, trainX, trainY, temperature = 1):
            """
            Input: Model, encoder, trainX, trainY, temperature
            Calculates the entropy and log_gmm_density for mnist, fashion_mnist and ambiguous_mnist
            Returns entropy, log_gmm_density for mnist, fashion_mnist and ambiguous_mnist
            """ 
            #Load Datasets
            #mnist
            eval_testX_mnist = load_datasets("mnist")
            #fashion_mnis
            eval_testX_fashion_mnist = load_datasets("fashion_mnist")
            #ambiguous_mnist
            eval_testX_ambiguous_mnist = load_datasets("ambiguous_mnist")
        
            #Get Features of dirty_mnist and init DDU
            features_dirty_mnist = encoder.predict(trainX)
            ddu_dirty_mnist_mnist = uncertainty.DDU(features_dirty_mnist, trainY)
            ddu_dirty_mnist_fashion_mnist = uncertainty.DDU(features_dirty_mnist, trainY)
            ddu_dirty_mnist_ambiguous_mnist = uncertainty.DDU(features_dirty_mnist, trainY)

            #mnist  
            entropy_mnist, log_gmm_density_mnist = get_entropy_log_gmm_density(eval_testX_mnist, model, encoder, ddu_dirty_mnist_mnist, temperature)

            #fashion_mnist
            entropy_fashion_mnist, log_gmm_density_fashion_mnist = get_entropy_log_gmm_density(eval_testX_fashion_mnist, model, encoder, ddu_dirty_mnist_fashion_mnist, temperature)

            #ambiguous_mnist
            entropy_ambiguous_mnist, log_gmm_density_ambiguous_mnist = get_entropy_log_gmm_density(eval_testX_ambiguous_mnist, model, encoder, ddu_dirty_mnist_ambiguous_mnist, temperature)

            return entropy_mnist, log_gmm_density_mnist, entropy_fashion_mnist, log_gmm_density_fashion_mnist,entropy_ambiguous_mnist, log_gmm_density_ambiguous_mnist

    def create_fig_1(mode):
        """
        Trains model and encoder
        Calculates entropy and log_gmm_density
        Plots results as in Figure 1

        mode:
        - train: just trains the model
        - calculate: Calculates entropy and log_gmm_density
        - plot: Plots results from saved entropy and log_gmm_density
        - calculate_and_plot: calculate and plot
        - all: do all above
        """

        print("Running in", mode, "mode")

        if mode in ["train", "calculate", "calculate_and_plot", "all"]:

            #Load DirtyMNIST dataset
            ds_train = ddu_dirty_mnist.DirtyMNIST("~/datasets", train=True, download=True) 

            #Create arrays of Dirty_Mnist dataset
            trainX, trainY, valX, valY = create_arrays(ds_train, temperature_scaling=True)

            print("Got train and validation arrays")

            #Load weights
            if mode in ["train", "all"]:
                #Train model and save weights
                model_path, encoder_path = train_model_function(trainX, trainY)

        if mode in ["calculate", "calculate_and_plot", "all"]: 
        
            # Define model and encoder path

            if(train_modBlock):
                if(train_ablate):
                    model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
                else: 
                    model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
            else:
                if(train_ablate):
                    model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
                else: 
                    model_path = 'results/full_models_afterTraining/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"


            if(train_modBlock):
                if(train_ablate):
                    encoder_path = 'results/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
                else: 
                    encoder_path = 'results/encoders/training_'+train_model+"_"+"SN"+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"
            else:
                if(train_ablate):
                    encoder_path = 'results/encoders/training_'+train_model+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+"/checkpoint"
                else: 
                    encoder_path = 'results/encoders/training_'+train_model+"_"+dataset+"_n_run_"+str(n_run)+"/checkpoint"

            print("Model Path", model_path)
            print("Encoder Path", encoder_path)

            #Load Weights
            model, encoder = load_model_and_encoder_weights(model_path, encoder_path)

            print("Got model and encoder")

            #Temperature Scaling
            temps = []  
            if(temperature_scaling):
                # perform temperature scaling to calibrate network
                temp_step = 0.1
                num_temps = 100
                T = 0.1
                opt_temp = T
                opt_nll = 1000000
                opt_ece = 1000000     

                for i in range(num_temps):
                    temp_logits = model.predict(valX, batch_size=batch_size)/T
                    if(temp_criterion == 'ece'):
                        ece_temp = 100*tfp.stats.expected_calibration_error(num_bins = 10, logits=temp_logits, labels_true=valY).numpy().astype(np.float32)
                        print("ECE: ", ece_temp, "Temp: ", T)
                        if(ece_temp < opt_ece):
                            print("Lower ECE found", "ece_temp: ", ece_temp, "opt_ece: ", opt_ece, "Temp: ", T)
                            opt_ece = ece_temp
                            opt_temp = T
                    elif(temp_criterion == 'nll'):
                        nll_temp = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valY, logits=temp_logits)).numpy().astype(np.float32)    
                        if(nll_temp < opt_nll):
                            opt_nll = nll_temp
                            opt_temp = T
                    else: 
                        nll_temp = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valY, logits=temp_logits)).numpy().astype(np.float32)    
                        if(nll_temp < opt_nll):
                            opt_nll = nll_temp
                            opt_temp = T
                    T += temp_step
                temp = opt_temp
                temps.append(temp)
                print("Optimal Temperature: %f with optimal NLL: %f and optimal ECE: %f"%(temp, opt_nll, opt_ece))

            #Calculate Entropies and log_gmm_density
            entropy_mnist, log_gmm_density_mnist, entropy_fashion_mnist, log_gmm_density_fashion_mnist, entropy_ambiguous_mnist, log_gmm_density_ambiguous_mnist = get_all_entropy_log_gmm_density(model, encoder, trainX, trainY, temperature = temp)

            print("Got entropy and density")
            
            #Save results
            np.savez('results/entropies_and_log_gmm_density'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+".npz", array1=entropy_mnist, array2=log_gmm_density_mnist, array3=entropy_fashion_mnist, array4=log_gmm_density_fashion_mnist, array5=entropy_ambiguous_mnist, array6=log_gmm_density_ambiguous_mnist)
            print("Results saved to path", 'results/entropies_and_log_gmm_density'+train_model+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+".npz")
            
        if mode in ["plot", "calculate_and_plot", "all"]:
            
            #Define class results to store results
            class results:
                def __init__(self, entropy_mnist, log_gmm_density_mnist, entropy_fashion_mnist, log_gmm_density_fashion_mnist, entropy_ambiguous_mnist, log_gmm_density_ambiguous_mnist):
                    #Entropy
                    self.entropy_mnist = entropy_mnist
                    self.entropy_fashion_mnist = entropy_fashion_mnist
                    self.entropy_ambiguous_mnist = entropy_ambiguous_mnist
                    
                    #Density
                    self.log_gmm_density_mnist = log_gmm_density_mnist  
                    self.log_gmm_density_fashion_mnist = log_gmm_density_fashion_mnist  
                    self.log_gmm_density_ambiguous_mnist = log_gmm_density_ambiguous_mnist
            
            #Load data
            loaded_data_LeNet = np.load('results/entropies_and_log_gmm_density'+"lenet"+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+".npz")
            loaded_data_VGG16 = np.load('results/entropies_and_log_gmm_density'+"vgg_16"+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+".npz")
            loaded_data_resnet = np.load('results/entropies_and_log_gmm_density'+"resnet"+"_"+"SN"+"_"+dataset+"_ablation"+"_n_run_"+str(n_run)+".npz")

            for idx, loaded_data in enumerate([loaded_data_LeNet, loaded_data_VGG16, loaded_data_resnet]): 
                entropy_mnist = loaded_data['array1']
                log_gmm_density_mnist = loaded_data['array2']
                entropy_fashion_mnist = loaded_data['array3']
                log_gmm_density_fashion_mnist = loaded_data['array4']
                entropy_ambiguous_mnist = loaded_data['array5']
                log_gmm_density_ambiguous_mnist = loaded_data['array6']

                if idx == 0:
                    results_LeNet = results(entropy_mnist, -log_gmm_density_mnist, entropy_fashion_mnist, -log_gmm_density_fashion_mnist, entropy_ambiguous_mnist, -log_gmm_density_ambiguous_mnist)
                if idx == 1:
                    results_VGG16 = results(entropy_mnist, -log_gmm_density_mnist, entropy_fashion_mnist, -log_gmm_density_fashion_mnist, entropy_ambiguous_mnist, -log_gmm_density_ambiguous_mnist)
                if idx == 2:
                    results_resnet = results(entropy_mnist, -log_gmm_density_mnist, entropy_fashion_mnist, -log_gmm_density_fashion_mnist, entropy_ambiguous_mnist, -log_gmm_density_ambiguous_mnist)

            #Plot Entropy and Density
            plot_entropy(results_LeNet, results_VGG16, results_resnet)
            plot_density(results_LeNet, results_VGG16, results_resnet)

#Create figure 1 from scratch for all 3 models
models = ["lenet", "vgg_16", "resnet"]
for model in models:
    train_model = model

    #Set ablate and modBlock
    train_modBlock = False
    train_ablate = False
    if model == "resnet":
        train_modBlock = True
        train_ablate = False

    print("Training and calculating for model: ", train_model)
    #Train model
    mode = "train"
    create_fig_1(mode)
    #Calculate entropy and density
    mode = "calculate"
    create_fig_1(mode)

#Plot entropy and density for all models together
mode = "plot"
create_fig_1(mode)
