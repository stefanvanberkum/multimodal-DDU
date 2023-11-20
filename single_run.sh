#!/bin/bash

# A single run for all WRN models.

# WRN models:
# - WRN softmax: model_type=wrn, modBlock=false, ablate=false
# - WRN DDU: model_type=wrn, modBlock=true, ablate=false
# - WRN ensemble: model_type=wrn-ensemble, modBlock=false, ablate=false
# - WRN DDU ablation: model_type=wrn, modBlock=true, ablate=true

# ResNet models:
# - ResNet softmax: model_type=resnet, modBlock=false, ablate=false
# - ResNet DDU: model_type=resnet, modBlock=true, ablate=false
# - ResNet ensemble: model_type=resnet-ensemble, modBlock=false, ablate=false
# - ResNet DDU ablation: model_type=resnet, modBlock=true, ablate=true

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

python train_models.py --model $model_type --dataset $dataset --modBlock --ablate --n_run $run
wait

