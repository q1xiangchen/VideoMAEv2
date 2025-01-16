"""Visualization of the reconstruction"""
import argparse
import os

import numpy as np
import torch
import random
from functools import partial
from run_mae_pretraining import get_args, get_model

import models  # noqa: F401
import utils
from einops import rearrange
from dataset import build_pretraining_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import multiple_pretrain_samples_collate
from models.motion_modulation import MotionLayer


import torch
from einops import rearrange

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

device = torch.device("cpu")
mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[:, None, None, None]
std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[:, None, None, None]


def visualize_frames(original, reconstructed, labels, combine, output_dir):
    b, c, t, h, w = original.shape
    original = rearrange(original, 'b c t h w -> b t c h w')
    labels = rearrange(labels, 'b c t h w -> b t c h w')
    reconstructed = rearrange(reconstructed, 'b c t h w -> b t c h w')
    combine = rearrange(combine, 'b c t h w -> b t c h w')
    for i in range(b):
        for j in range(t):
            origin_img = original[i, j]
            masked_img = labels[i, j]
            recon_img = reconstructed[i, j]
            combine_img = combine[i, j]

            orig_img = ToPILImage()(origin_img)
            masked_img = ToPILImage()(masked_img)
            recon_img = ToPILImage()(recon_img)
            combine_img = ToPILImage()(combine_img)
            
            # Display the frames (optional)
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            axes[0].imshow(orig_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            axes[1].imshow(masked_img)
            axes[1].set_title("Masked")
            axes[1].axis('off')
            axes[2].imshow(recon_img)
            axes[2].set_title("Reconstructed")
            axes[2].axis('off')
            axes[3].imshow(combine_img)
            axes[3].set_title("Reconstructed+Original")
            axes[3].axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparison_b{i}_t{j}.png")
            plt.close()


def reconstruct_frames(original_frames, reconstructed_patches, decoder_mask, labels, patch_size=(2, 16, 16)):
    # check the range
    print("Original frames range: ", original_frames.min(), original_frames.max())
    print("Reconstructed patches range: ", reconstructed_patches.min(), reconstructed_patches.max())
    print("gt patches range: ", labels.min(), labels.max())

    p0, p1, p2 = patch_size
    # Reshape original frames into patches
    original_patches = rearrange(
        original_frames,
        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
        p0=p0, p1=p1, p2=p2
    )
    B, N, C = original_patches.shape
    
    # Combine original and reconstructed patches
    combined_frames = original_patches.clone()
    reconstructed_frames = torch.zeros_like(original_patches)
    masked_frames = torch.zeros_like(original_patches)
    for i in range(B):
        reconstruct_j = 0
        for j in range(N):
            if not decoder_mask[i, j]:
                combined_frames[i, j] = reconstructed_patches[i, reconstruct_j]
                reconstructed_frames[i, j] = reconstructed_patches[i, reconstruct_j]
                masked_frames[i, j] = labels[i, reconstruct_j]
                reconstruct_j += 1
    
    # Reshape patches back into full frames
    combined_frames = rearrange(
        combined_frames,
        'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
        p0=p0, p1=p1, p2=p2, t=8, h=14, w=14
    )
    
    reconstructed_frames = rearrange(
        reconstructed_frames,
        'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
        p0=p0, p1=p1, p2=p2, t=8, h=14, w=14
    )

    labels = rearrange(
        masked_frames,
        'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
        p0=p0, p1=p1, p2=p2, t=8, h=14, w=14
    )
    
    return reconstructed_frames, labels, combined_frames


def visualize_reconstruction(args):
    # prepare the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # get the model
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print("Load ckpt from %s" % args.checkpoint)
    checkpoint_model = None
    for model_key in ['model', 'module', 'state_dict']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            # replace the key name of checkpoint_model
            checkpoint_model = {
                ".".join(k.split(".")[1:]) if "_orig_mod" in k else k: v
                for k, v in checkpoint_model.items()
                }
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    utils.load_state_dict(model, checkpoint_model)
    model.eval()

    # get dataset
    dataset_test = build_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_test))
    
    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True)
    
    print("Start visualization...")
    for idx, batch in enumerate(data_loader_train):
        # get the data
        images, bool_masked_pos, decode_masked_pos = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                     None,
                                                                     None,
                                                                     None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                   None, None,
                                                                   None]
            unnorm_images = images * std + mean  # in [0, 1]

            if unnorm_images.shape[2] == 17:
                unnorm_images = unnorm_images[:, :, 1:, :, :]

            if args.normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size[0],
                    p2=patch_size[1])
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_squeeze_mean = images_squeeze.mean(dim=-2, keepdim=True)
                images_squeeze_std = images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                images_squeeze_mean = rearrange(images_squeeze_mean, 'b n p c -> b n (p c)')
                images_squeeze_std = rearrange(images_squeeze_std, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size[0],
                    p2=patch_size[1])


            B, N, C = images_patch.shape
            images_squeeze_mean = images_squeeze_mean[~decode_masked_pos].reshape(B, -1, 3)
            images_squeeze_std = images_squeeze_std[~decode_masked_pos].reshape(B, -1, 3)

            # duplicate the last dim C/3 times for images_squeeze_mean and images_squeeze_std
            images_squeeze_mean = images_squeeze_mean.repeat(1, 1, C//3)
            images_squeeze_std = images_squeeze_std.repeat(1, 1, C//3)

            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)
            labels = labels * images_squeeze_std + images_squeeze_mean

            outputs, _ = model(images, bool_masked_pos, decode_masked_pos)
            outputs = outputs * images_squeeze_std + images_squeeze_mean

        if args.motion_layer != "baseline":
            motion_layer = model.motion_layer
            unnorm_images, _ = motion_layer(images)
            unnorm_images = unnorm_images * std + mean

        # convert the outputs and gt to the original shape
        outputs, labels, combine = reconstruct_frames(unnorm_images, outputs, decode_masked_pos, labels, patch_size=(2,16,16))
        visualize_frames(unnorm_images, outputs, labels, combine, args.output_dir)

        print("Finish visualization.")
        break


if __name__ == '__main__':
    seed = 7234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint_dict = {
        "baseline": "model_zoo/vit_b_sthv2_800e_base.pth",
        "finetune_w_layer_10_4_100_fixed": "model_zoo/vit_b_sthv2_800e_fix_h10_w10_t4_checkpoint-199.pth",
        "finetune_w_layer_150_12_100_fixed": "model_zoo/vit_b_sthv2_800e_fix_h150_w150_t12_checkpoint-199.pth",
    }

    for motion_layer in ["baseline", "finetune_w_layer_10_4_100_fixed", "finetune_w_layer_150_12_100_fixed"]:
        args = get_args()
        args.model = 'pretrain_videomae_base_patch16_224'
        args.data_path = "data/decoder_samples.csv"
        args.sample_rate = 2
        args.output_dir = "visual/decoder_visualization/" 

        args.mask_type = "tube"
        args.normlize_target = True
        args.mask_ratio = 0.9
        args.decoder_mask_type = "run_cell"
        args.decoder_mask_ratio = 0.5
        args.decoder_depth = 4
        args.with_checkpoint = True
        args.num_frames = 16
        args.num_sample = 1
        args.num_workers = 1
        args.batch_size = 4
        args.motion_layer = motion_layer
        args.checkpoint = checkpoint_dict[motion_layer]
        args.output_dir += motion_layer

        visualize_reconstruction(args)