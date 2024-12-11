import os
import time

def generate_script(
    save_dir,
    split=1,
    output_dir='./work_dir/default_job',
    data_path='./data/hmdb51',
    model_path='./model_zoo/vit_g_hybrid_pt_1200e.pth',
    job_name="default_job",
    partition="video",
    n_nodes=1,
    gpus=4,
    gpus_per_node=4,
    model_name="vit_base_patch16_224",
    batch_size=3,
    num_sample=2,
    input_size=224,
    short_side_size=224,
    save_ckpt_freq=10,
    num_frames=16,
    sampling_rate=2,
    optimizer="adamw",
    learning_rate=5e-4,
    layer_decay=0.90,
    num_workers=10,
    opt_betas=(0.9, 0.999),
    weight_decay=0.05,
    epochs=25,
    drop_path=0.35,
    head_drop_rate=0.5,
    test_num_segment=5,
    test_num_crop=3,
    dist_eval=True,
    enable_deepspeed=True,
    extra_args="",
    motion_layer="baseline",
    end_to_end=True,
):
    script = f"""
#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1


OUTPUT_DIR='{output_dir}'
DATA_PATH='{data_path}_{split}'
MODEL_PATH='{model_path}'

PARTITION={partition}
# 8 for 1 node, 16 for 2 node, etc.
N_NODES={n_nodes}  # Number of nodes
GPUS={gpus}
GPUS_PER_NODE={gpus_per_node}
SRUN_ARGS=${{SRUN_ARGS:-""}}
PY_ARGS=${{@:2}}

# batch_size can be adjusted according to the graphics card
torchrun --nproc_per_node=${{GPUS_PER_NODE}} \\
        --master_port ${{MASTER_PORT}} --nnodes=${{N_NODES}} \\
        run_class_finetuning.py \\
        --model {model_name} \\
        --data_set HMDB51 \\
        --nb_classes 51 \\
        --data_path ${{DATA_PATH}} \\
        --finetune ${{MODEL_PATH}} \\
        --log_dir ${{OUTPUT_DIR}} \\
        --output_dir ${{OUTPUT_DIR}} \\
        --batch_size {batch_size} \\
        --num_sample {num_sample} \\
        --input_size {input_size} \\
        --short_side_size {short_side_size} \\
        --save_ckpt_freq {save_ckpt_freq} \\
        --num_frames {num_frames} \\
        --sampling_rate {sampling_rate} \\
        --opt {optimizer} \\
        --lr {learning_rate} \\
        --layer_decay {layer_decay} \\
        --num_workers {num_workers} \\
        --opt_betas {opt_betas[0]} {opt_betas[1]} \\
        --weight_decay {weight_decay} \\
        --epochs {epochs} \\
        --drop_path {drop_path} \\
        --head_drop_rate {head_drop_rate} \\
        --test_num_segment {test_num_segment} \\
        --test_num_crop {test_num_crop} \\
        --motion_layer {motion_layer} \\
        --end_to_end {end_to_end} \\
        {"--dist_eval " if dist_eval else ""} \\
        {"--enable_deepspeed " if enable_deepspeed else ""} \\
        {extra_args} \\
"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{job_name}.sh"), "w") as f:
        f.write(script)
    print(f"\n{'config script saved at':<25}{save_dir}/{job_name}.sh")


def job_script(config_path, save_path):
    script = f"""
#!/bin/bash
#PBS -P cp23
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=380GB
#PBS -q gpuvolta
#PBS -l jobfs=32GB
#PBS -l walltime=12:00:00
#PBS -l wd
#PBS -l storage=scratch/dg97+scratch/kf09+gdata/kf09

cd /home/135/qc2666/dg/VideoMAEv2

module load cuda/12.2.2

# Activate Conda
export CONDA_ENV='/scratch/kf09/qc2666/miniconda3/bin/activate'
source $CONDA_ENV videomae

bash {config_path}
"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(script)
    print(f"{'Job script saved at':<25}{save_path}")

if __name__ == "__main__":
    for pretrain_epochs in [60, 100]:
        for finetune_split in [1]:
            vit_model = "vit_b"

            # model_name = "vit_giant_patch14_224"
            model_name = "vit_base_patch16_224"

            pretrain_dataset = "k710"
            # pretrain_dataset = "hybrid"
            
            finetune_dataset = "hmdb51"
            motion_layer = "finetune_w_layer"
            end_to_end = True
            
            pth_name = f"{vit_model}_{pretrain_dataset}_pt_{pretrain_epochs}e"
            pth_name += "_w_layer" if motion_layer == "pretrain_w_layer" else ""
            model_path = f"./model_zoo/{pth_name}.pth"
            
            job_name = f"{pth_name}_{finetune_dataset}_{finetune_split}_ft"
            job_name += "_w_layer" if motion_layer != "baseline" else ""
            job_name += "_e2e" if end_to_end else "_freeze"
            
            generate_script(
                save_dir=os.path.join("./auto_script", time.strftime("%m%d")),
                split=finetune_split,
                output_dir=f"./work_dir/{job_name}",
                data_path=f"./data/{finetune_dataset}",
                model_path=model_path,
                job_name=job_name,
                model_name=model_name,
                batch_size=2 if vit_model == "vit_b" else 1,
                motion_layer=motion_layer,
                end_to_end=end_to_end,
            )

            job_script(
                config_path=f"./scripts/auto_script/{time.strftime('%m%d')}/{job_name}.sh",
                save_path=f"./job_script/{job_name}.sh"
            )

            print(f"{'  Job status  ':*^45}")
            print('{:>20}'.format("Pretrain model:"), f"{vit_model}_{pretrain_dataset}_pt_{pretrain_epochs}e.pth")
            print('{:>20}'.format("Finetune dataset:"), f"{finetune_dataset}_{finetune_split}")
            print('*' * 45)
            input("Press Enter to submit the job")

            os.system(f"qsub ./job_script/{job_name}.sh")