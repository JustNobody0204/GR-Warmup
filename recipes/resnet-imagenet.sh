#!/bin/bash

### Set you would like to use. If not set, all the gpus will be used by default.
# export CUDA_VISIBLE_DEVICES=0,1,2,3

ENV_PATH=your-path-to-your-python-venv
source $ENV_PATH

SRC_PATH=your-path-to-this-code-repo
### Could be ../../ when you run bash command at this folder.
cd $SRC_PATH

### The training config py file.
TRAIN_CONFIG_DIR=your-path-to-your-training-config-file.py
## Could be gnp/config/train_config.py

### The output dir where your would like to store your results.
WORKING_DIR=your-path-to-your-output-dir

MODEL_NAME=ResNet50

INIT_SEEDS=0
SEEDS=0
BATCH_SIZE=512
BASE_LR=0.2
TOTAL_EPOCH=100
L2_REG=1e-4

DATASET_NAME=imagenet
DATASET_IMAGE_LEVEL_AUG=none
DATASET_BATCH_LEVEL_AUG=none

OPT_TYPE=Momentum
GNP_R=0.05
GNP_ALPHA=0.8

python3 -m gnp.main.main --config=${TRAIN_CONFIG_DIR} \
                         --working_dir=${WORKING_DIR} \
                         --config.init_seeds=$INIT_SEEDS \
                         --config.seeds=$SEEDS \
                         --config.batch_size=$BATCH_SIZE \
                         --config.base_lr=$BASE_LR \
                         --config.l2_regularization=$L2_REG \
                         --config.total_epochs=$TOTAL_EPOCH \
                         --config.dataset.dataset_name=$DATASET_NAME \
                         --config.dataset.batch_level_augmentations=$DATASET_BATCH_LEVEL_AUG \
                         --config.dataset.image_level_augmentations=$DATASET_IMAGE_LEVEL_AUG \
                         --config.opt.opt_type=$OPT_TYPE \
                         --config.gnp.alpha=$GNP_ALPHA \
                         --config.gnp.r=$GNP_R \
                         --config.model.model_name=$MODEL_NAME \

