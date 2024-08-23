#! bin/bash

GPU=$1
NUM_GPU=$(echo $GPU | tr ',' ' ' | wc -w)

CUDA_VISIBLE_DEVICES=$GPU tools/dist_train.sh \
 projects/configs/bevformer/bevformer_bevrestore_lr_bev.py \
 $NUM_GPU --no-validate

CUDA_VISIBLE_DEVICES=$GPU tools/dist_train.sh \
 projects/configs/bevformer/bevformer_bevrestore_hr_bev.py \
 $NUM_GPU --no-validate
