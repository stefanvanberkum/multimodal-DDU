""" Code for plotting entropies in OoD-detection """
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# set plot parameters
bin_num = 30
stat = "probability"
alpha = 0.6
binrange = [[-250, 100], [1000, 3000], [-1000, 2500]]

# load data
loaded_data = np.load('densities_and_entropies/wrn_SN_cifar10_vs_svhn.npz')
all_epistemics_in = loaded_data['array1']
all_epistemics_out = loaded_data['array2']
all_aleatorics_in = loaded_data['array3']
all_aleatorics_out = loaded_data['array4']

# get mean scores
epistemics_in_mean = np.mean(all_epistemics_in, axis=0)
epistemics_out_mean = np.mean(all_epistemics_out, axis=0)
aleatorics_in_mean = np.mean(all_aleatorics_in, axis=0)
aleatorics_out_mean = np.mean(all_aleatorics_out, axis=0)


#Create figure with 500 dpi
sb.set()
plt.figure(dpi=500)
            
# train_models = [results_1, results_2, results_3]
# model_names = ["LeNet", "VGG16", "ResNet18"]

#create subplots for LeNet, VGG16, and ResNet18
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6))

# fig, ax = plt.subplot(111)
# combined_data = np.concatenate((train_model_result.log_gmm_density_mnist, train_model_result.log_gmm_density_ambiguous_mnist))
sb.histplot(data = aleatorics_in_mean, stat=stat, bins = bin_num, label = "iD (Cifar10)", alpha = alpha, element = "step", edgecolor = "k", color = "dodgerblue")
sb.histplot(data = aleatorics_out_mean, stat=stat, bins = bin_num, label =  "OoD (SVHN)", alpha = alpha, element = "step",edgecolor = "k", color = "darkorange")

# plt.set_xlabel("Density")
# plt.set_ylabel("Fraction")

legend_1 = plt.legend(loc = "upper left")
# # legend_2 = ax2.legend(loc = "upper left")
# # legend_3 = ax3.legend(loc = "upper left")
        
# legend_1.get_texts()[1].set_text('Dirty Mnist')
# legend_1.get_texts()[0].set_text('Ambiguous Mnist')

# legend_2.get_texts()[1].set_text('Dirty Mnist')
# legend_2.get_texts()[0].set_text('Ambiguous Mnist')

# legend_3.get_texts()[1].set_text('Dirty Mnist')
# legend_3.get_texts()[0].set_text('Ambiguous Mnist')

#save figure
plt.savefig("entropy_cifar10_vs_svhn")

plt.show()