#!/bin/bash

# A single run for all WRN models.

# WRN models:
# - WRN softmax: model_type=wrn, modBlock=False, ablate=False
# - WRN DDU: model_type=wrn, modBlock=True, ablate=False
# - WRN ensemble: model_type=wrn-ensemble, modBlock=False, ablate=False
# - WRN DDU ablation: model_type=wrn, modBlock=True, ablate=True

# ResNet models:
# - ResNet softmax: model_type=resnet, modBlock=False, ablate=False
# - ResNet DDU: model_type=resnet, modBlock=True, ablate=False
# - ResNet ensemble: model_type=resnet-ensemble, modBlock=False, ablate=False
# - ResNet DDU ablation: model_type=resnet, modBlock=True, ablate=True

run=1
dataset=cifar10

# WRN softmax.
model_type=wrn

python train_models.py --model $model_type --dataset $dataset --n_run $run
wait

# WRN DDU.
model_type=wrn

python train_models.py --model $model_type --dataset $dataset --modBlock --n_run $run
wait

# WRN ensemble.
model_type=wrn-ensemble

python train_models.py --model $model_type --dataset $dataset --n_run $run
wait

# WRN DDU ablation.
model_type=wrn
modBlock=True

python train_models.py --model $model_type --dataset $dataset --modBlock --ablate --n_run $run
wait

