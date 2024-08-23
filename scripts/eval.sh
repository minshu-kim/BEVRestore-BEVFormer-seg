#! bin/bash

CUDA_VISIBLE_DEVICES=0 tools/dist_test.sh \
 projects/configs/bevformer/bevformer_small_seg.py \
 work_dirs/bevformer_small_seg/epoch_24.pth 1
