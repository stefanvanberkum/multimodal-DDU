#!/bin/bash

# Five runs on CIFAR-10 and CIFAR-100 for one model.

# WRN models:
# - WRN softmax: model_type=wrn, modblock=False, ablate=False
# - WRN DDU: model_type=wrn, modblock=True, ablate=False
# - WRN ensemble: model_type=wrn-ensemble, modblock=False, ablate=False
# - WRN DDU ablation: model_type=wrn, modblock=True, ablate=True

# ResNet models:
# - ResNet softmax: model_type=resnet, modblock=False, ablate=False
# - ResNet DDU: model_type=resnet, modblock=True, ablate=False
# - ResNet ensemble: model_type=resnet-ensemble, modblock=False, ablate=False
# - ResNet DDU ablation: model_type=resnet, modblock=True, ablate=True

model_type = wrn
modblock = True
ablate = False

for run in {1..5}
do
	for dataset in cifar10 cifar100
	do
		python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
		wait
	done
done

