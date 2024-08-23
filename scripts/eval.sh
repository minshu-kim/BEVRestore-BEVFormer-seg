#! bin/bash

CUDA_VISIBLE_DEVICES=0 tools/dist_test.sh \
 projects/configs/bevformer/bevformer_bevrestore_hr_bev.py \
 work_dirs/bevformer_bevrestore_hr_bev/epoch_24.pth 1
