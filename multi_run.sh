#!/bin/bash

# Five runs on CIFAR-10 and CIFAR-100 for one model.

# WRN models:
# - WRN softmax: MODEL_TYPE=wrn, MODBLOCK=False, ABLATE=False
# - WRN DDU: MODEL_TYPE=wrn, MODBLOCK=True, ABLATE=False
# - WRN ensemble: MODEL_TYPE=wrn-ensemble, MODBLOCK=False, ABLATE=False
# - WRN DDU ablation: MODEL_TYPE=wrn, MODBLOCK=True, ABLATE=True

# ResNet models:
# - ResNet softmax: MODEL_TYPE=resnet, MODBLOCK=False, ABLATE=False
# - ResNet DDU: MODEL_TYPE=resnet, MODBLOCK=True, ABLATE=False
# - ResNet ensemble: MODEL_TYPE=resnet-ensemble, MODBLOCK=False, ABLATE=False
# - ResNet DDU ablation: MODEL_TYPE=resnet, MODBLOCK=True, ABLATE=True

MODEL_TYPE = wrn
MODBLOCK = True
ABLATE = False

for RUN in {1..5}
do
	for DATASET in cifar10 cifar100
	do
		python train_models.py --model $MODEL_TYPE --dataset $DATASET --modblock $MODBLOCK --ablate $ABLATE --n_run $RUN
		wait
	done
done

