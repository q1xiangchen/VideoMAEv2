#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# official train/test splits. valid numbers: 1, 2, 3
SPLIT=${SPLIT:-1}

OUTPUT_DIR='./work_dir/vit_b_k710_pt_100e_hmdb51_'${SPLIT}'_ft'
DATA_PATH='./data/hmdb51_'${SPLIT}
MODEL_PATH='./model_zoo/vit_b_k710_pt_100e.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
N_NODES=${N_NODES:-1}  # Number of nodes
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
torchrun --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set HMDB51 \
        --nb_classes 51 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 3 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --lr 5e-4 \
        --layer_decay 0.90 \
        --num_workers 10 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 25 \
        --drop_path 0.35 \
        --head_drop_rate 0.5 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval \
        --enable_deepspeed \
        ${PY_ARGS}