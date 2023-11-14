#!/bin/bash

# Five runs on CIFAR-10 and CIFAR-100 for one model.

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

model_type=wrn
modBlock=true
ablate=false

for run in {1..5}
do
  for dataset in cifar10 cifar100
  do
    if $modBlock
    then
      if $ablate
      then
        python train_models.py --model $model_type --dataset $dataset --modBlock --ablate --n_run $run
      else
        python train_models.py --model $model_type --dataset $dataset --modBlock --n_run $run
      fi
    else
      if $ablate
      then
        python train_models.py --model $model_type --dataset $dataset --ablate --n_run $run
      else
        python train_models.py --model $model_type --dataset $dataset --n_run $run
      fi
    fi
    wait
  done
done
