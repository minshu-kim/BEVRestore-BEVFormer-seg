GPU=$1
NUM_GPU=$(echo $GPU | tr ',' ' ' | wc -w)

CUDA_VISIBLE_DEVICES=$GPU tools/dist_train.sh \
 projects/configs/bevformer/bevformer_seg.py \
 $NUM_GPU --no-validate
