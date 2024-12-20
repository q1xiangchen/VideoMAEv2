#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1


OUTPUT_DIR='./results/sthv2/vit_b_sthv2_pt_200e_base_ft_w_0_param'
DATA_PATH='./data/sthv2'
MODEL_PATH='./model_zoo/vit_b_sthv2_800e_base.pth'

# 8 for 1 node, 16 for 2 node, etc.
N_NODES=1  # Number of nodes
GPUS=4
GPUS_PER_NODE=4
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
torchrun --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set SSV2 \
        --nb_classes 174 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 2e-3 \
        --num_workers 10 \
        --layer_decay 0.65 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 25 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --motion_layer zero_param \
        --dist_eval  \
        --end_to_end \
