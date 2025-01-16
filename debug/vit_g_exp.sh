#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

OUTPUT_DIR='./visual/vit_g_exp_baseline'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='./data/decoder_samples.csv'  # The data list file path.

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_mae_pretraining.py` for the meaning of the following hyperreferences
torchrun --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 2 \
        --with_checkpoint \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 1 \
        --num_workers 1 \
        --opt adamw \
        --lr 6e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 5 \
        --epochs 300 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --motion_layer baseline \
        --finetune model_zoo/vit_b_hybrid_pt_1200e.bin \
