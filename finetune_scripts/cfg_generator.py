import os
import time


## Configurations section
dict_dataset_path = {
    "HMDB51": "./data/hmdb51_1",
    "UCF101": "./data/ucf101_1",
    "Kinetics-400": "PATH_TO_KINETICS400",
    "Kinetics-600": "PATH_TO_KINETICS600",
    "Kinetics-700": "PATH_TO_KINETICS700",
    "SSV2": "PATH_TO_SSV2",
}

num_node = 1
num_gpu_per_node = 4
num_gpus = num_node * num_gpu_per_node

pt_model_path = "./model_zoo/vit_b_hybrid_pt_1200e.bin"
## Configurations section


dict_dataset = {
    "HMDB51": 51,
    "UCF101": 101,
    "Kinetics-400": 400,
    "Kinetics-600": 600,
    "Kinetics-700": 700,
    "SSV2": 174,
}

dict_lr = {
    "HMDB51": 5e-4,
    "UCF101": 1e-3,
    "Kinetics-400": 1e-5,
    "Kinetics-600": 1e-5,
    "Kinetics-700": 1e-5,
    "SSV2": 3e-4,
}

dict_warmup_epoch = {
    "HMDB51": 5,
    "UCF101": 5,
    "Kinetics-400": 0,
    "Kinetics-600": 0,
    "Kinetics-700": 0,
    "SSV2": 5,
}

dict_epoch = {
    "HMDB51": 15,
    "UCF101": 50,
    "Kinetics-400": 3,
    "Kinetics-600": 3,
    "Kinetics-700": 3,  
    "SSV2": 10,
}

dict_drop_path = {
    "HMDB51": 0.35,
    "UCF101": 0.35,
    "Kinetics-400": 0.3,
    "Kinetics-600": 0.3,
    "Kinetics-700": 0.3,
    "SSV2": 0.35,
}

dict_bs = {
    "HMDB51": 24,
    "UCF101": 24,
    "Kinetics-400": 32,
    "Kinetics-600": 32,
    "Kinetics-700": 32,
    "SSV2": 96,
}

dict_sampling_rate = {
    "HMDB51": 2,
    "UCF101": 4,
    "Kinetics-400": 4,
    "Kinetics-600": 4,
    "Kinetics-700": 4,
    "SSV2": 4,
}


def calculate_accum_steps(original_total_batch_size, num_gpus, bs_max):
    new_total_batch_size = num_gpus * bs_max
    accum_steps = original_total_batch_size // new_total_batch_size

    if original_total_batch_size % new_total_batch_size != 0:
        print(
            f"Warning: The original total batch size ({original_total_batch_size}) "
            f"is not evenly divisible by the new batch size per iteration ({new_total_batch_size})."
        )

    return accum_steps



def gen_config(result_dir, pt_model_path, motion_layer, save_path, dataset, num_gpus):
    batch_size = 3 if dataset == "HMDB51" or dataset == "UCF101" else 4
    update_freq = calculate_accum_steps(dict_bs[dataset], num_gpus, batch_size)
    config = f"""
#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='./finetune_results/{dataset}/{result_dir}'
DATA_PATH='{dict_dataset_path[dataset]}'
MODEL_PATH='{pt_model_path}'

N_NODES={num_node}
GPUS_PER_NODE={num_gpu_per_node}
SRUN_ARGS=${{SRUN_ARGS:-""}}
PY_ARGS=${{@:2}}

# batch_size can be adjusted according to the graphics card
torchrun --nproc_per_node=${{GPUS_PER_NODE}} \\
        --master_port ${{MASTER_PORT}} --nnodes=${{N_NODES}} \\
        run_class_finetuning.py \\
        --model vit_base_patch16_224 \\
        --data_set {dataset} \\
        --nb_classes {dict_dataset[dataset]} \\
        --data_path ${{DATA_PATH}} \\
        --finetune ${{MODEL_PATH}} \\
        --log_dir ${{OUTPUT_DIR}} \\
        --output_dir ${{OUTPUT_DIR}} \\
        --batch_size {batch_size} \\
        --num_sample 2 \\
        --input_size 224 \\
        --short_side_size 224 \\
        --save_ckpt_freq 10 \\
        --num_frames 16 \\
        --sampling_rate {dict_sampling_rate[dataset]} \\
        --opt adamw \\
        --lr {dict_lr[dataset]} \\
        --layer_decay 0.9 \\
        --num_workers 10 \\
        --opt_betas 0.9 0.999 \\
        --weight_decay 0.05 \\
        --epochs {dict_epoch[dataset]} \\
        --update_freq {update_freq} \\
        --warmup_epochs {dict_warmup_epoch[dataset]} \\
        --drop_path {dict_drop_path[dataset]} \\
        --head_drop_rate 0.5 \\
        --test_num_segment 5 \\
        --test_num_crop 3 \\
        --motion_layer {motion_layer} \\
        --dist_eval  \\
        --end_to_end \\
"""
    with open(save_path, 'w') as f:
        f.write(config) 


if __name__ == "__main__":
    # finetune_w_layer_10_2_100_fixed
    # spatial = 10, temporal = 2, slope = 100, state = fixed

    for dataset in ["Kinetics-400", "Kinetics-600", "Kinetics-700", "SSV2", "HMDB51", "UCF101"]:
        print(f"Dataset: {dataset}")
        count = False
        for spatial in [10, 112]:
            for state in ["fixed", "learn", "baseline"]:
                temporal = 2 if spatial == 10 else 8
                motion_layer = f"finetune_w_layer_{spatial}_{temporal}_100_{state}"
                
                if state == "baseline":
                    if count: continue
                    motion_layer = "baseline"
                    count = True
                    
                result_dir = f"{pt_model_path.split('/')[1].split('.')[0]}_{motion_layer}"
                cfg_save_path = f"./cfg/{dataset}/{time.strftime('%m%d')}/{result_dir}.sh"

                os.makedirs(os.path.dirname(cfg_save_path), exist_ok=True)
                gen_config(result_dir, pt_model_path, motion_layer, cfg_save_path, dataset, num_gpus)
                print(motion_layer)