#!/bin/bash

# A single run for all WRN models.

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

run = 1
dataset = cifar10

# WRN softmax.
model_type = wrn
modblock = False
ablate = False

python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
wait

# WRN DDU.
model_type = wrn
modblock = True
ablate = False

python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
wait

# WRN ensemble.
model_type = wrn-ensemble
modblock = False
ablate = False

python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
wait

# WRN DDU ablation.
model_type = wrn
modblock = True
ablate = True

python train_models.py --model $model_type --dataset $dataset --modblock $modblock --ablate $ablate --n_run $run
wait

