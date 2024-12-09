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
    extra_args=""
):
    script = f"""
#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1


OUTPUT_DIR='{output_dir}'
DATA_PATH='{data_path}_{split}'
MODEL_PATH='{model_path}'

JOB_NAME={job_name}
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
        --model vit_base_patch16_224 \\
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
        {"--dist_eval " if dist_eval else ""} \\
        {"--enable_deepspeed " if enable_deepspeed else ""} \\
        {extra_args} \\
"""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{job_name}.sh"), "w") as f:
        f.write(script)
    print(f"Script saved at {save_dir}/{job_name}.sh")


if __name__ == "__main__":
    vit_model = "vit_b"
    pretrain_dataset = "k710"
    pretrain_epochs = 100
    finetune_dataset = "hmdb51"
    finetune_split = 1

    job_name = f"{vit_model}_{pretrain_dataset}_pt_{pretrain_epochs}e_{finetune_dataset}_{finetune_split}_ft"

    generate_script(
        save_dir=os.path.join("./work_dir", time.strftime("%m%d")),
        split=finetune_split,
        output_dir=f"./work_dir/{job_name}",
        data_path=f"./data/{finetune_dataset}",
        model_path=f"./model_zoo/{vit_model}_{pretrain_dataset}_pt_{pretrain_epochs}e.pth",
        job_name=job_name,
    )
