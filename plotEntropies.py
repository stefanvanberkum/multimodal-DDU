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
            

# fig, ax = plt.subplot(111)
# combined_data = np.concatenate((train_model_result.log_gmm_density_mnist, train_model_result.log_gmm_density_ambiguous_mnist))
sb.histplot(data = aleatorics_in_mean, stat=stat, bins = bin_num, label = "iD (Cifar10)", alpha = alpha, element = "step", edgecolor = "k", color = "dodgerblue")
sb.histplot(data = aleatorics_out_mean, stat=stat, bins = bin_num, label =  "OoD (SVHN)", alpha = alpha, element = "step",edgecolor = "k", color = "darkorange")



legend_1 = plt.legend(loc = "upper left")

#save figure
plt.savefig("entropy_cifar10_vs_svhn")

plt.show()