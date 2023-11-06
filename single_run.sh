#!/bin/bash

# A single run for all WRN models.

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

RUN = 1
DATASET = cifar10

# WRN softmax.
MODEL_TYPE = wrn
MODBLOCK = False
ABLATE = False

python train_models.py --model $MODEL_TYPE --dataset $DATASET --modblock $MODBLOCK --ablate $ABLATE --n_run $RUN
wait

# WRN DDU.
MODEL_TYPE = wrn
MODBLOCK = True
ABLATE = False

python train_models.py --model $MODEL_TYPE --dataset $DATASET --modblock $MODBLOCK --ablate $ABLATE --n_run $RUN
wait

# WRN ensemble.
MODEL_TYPE = wrn-ensemble
MODBLOCK = False
ABLATE = False

python train_models.py --model $MODEL_TYPE --dataset $DATASET --modblock $MODBLOCK --ablate $ABLATE --n_run $RUN
wait

# WRN DDU ablation.
MODEL_TYPE = wrn
MODBLOCK = True
ABLATE = True

python train_models.py --model $MODEL_TYPE --dataset $DATASET --modblock $MODBLOCK --ablate $ABLATE --n_run $RUN
wait

