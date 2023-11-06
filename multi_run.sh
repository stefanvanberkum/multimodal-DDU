#!/bin/bash

# Five runs on CIFAR-10 and CIFAR-100 for one model.

# WRN models:
# - WRN softmax: model_type=wrn, modblock=0, ablate=0
# - WRN DDU: model_type=wrn, modblock=1, ablate=0
# - WRN ensemble: model_type=wrn-ensemble, modblock=0, ablate=0
# - WRN DDU ablation: model_type=wrn, modblock=1, ablate=1

# ResNet models:
# - ResNet softmax: model_type=resnet, modblock=0, ablate=0
# - ResNet DDU: model_type=resnet, modblock=1, ablate=0
# - ResNet ensemble: model_type=resnet-ensemble, modblock=0, ablate=0
# - ResNet DDU ablation: model_type=resnet, modblock=1, ablate=1

model_type=wrn
modblock=1
ablate=0

for run in {1..5}
do
	for dataset in cifar10 cifar100
	do
		python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
		wait
	done
done

