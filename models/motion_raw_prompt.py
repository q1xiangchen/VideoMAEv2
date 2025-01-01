import torch
import torch.nn as nn
# import wandb

def rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute([order.index(dim) for dim in "BTCHW"])


def reverse_rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute(["BTCHW".index(dim) for dim in order])


class MotionRawLayer(torch.nn.Module):
    def __init__(self, exp_name = "", penalty_weight=1.0):
        super(MotionRawLayer, self).__init__()
        # default configs
        self.input_permutation = "BCTHW"   # Input format from video reader
        self.input_color_order = "RGB"     # Input format from video reader
        self.gray_scale = {"B": 0.114, "G": 0.587, "R": 0.299}

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
    def forward(self, video_seq):
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        loss = 0
        
        # normalize the input tensor back to [0, 1]
        input_std = torch.tensor(self.input_std).view(1, 1, 3, 1, 1).to(video_seq.device).to(video_seq.dtype)
        input_mean = torch.tensor(self.input_mean).view(1, 1, 3, 1, 1).to(video_seq.device).to(video_seq.dtype)
        norm_seq = video_seq * input_std + input_mean

        # transfor the input tensor to grayscale 
        weights = torch.tensor([self.gray_scale[idx] for idx in self.input_color_order], 
                               dtype=norm_seq.dtype, device=norm_seq.device)
        grayscale_video_seq = torch.einsum("btcwh, c -> btwh", norm_seq, weights)

        ### frame difference ###
        frame_diff = grayscale_video_seq[:,1:] - grayscale_video_seq[:,:-1]
        # frame_diff = frame_diff.clamp(-1, 1)

        ### min-max norm ###
        norm_attention = ((frame_diff + 1) / 2).unsqueeze(2)
        pad_norm_attention = norm_attention.repeat(1, 1, 3, 1, 1)  

        return reverse_rearrange_tensor((pad_norm_attention * video_seq[:,1:]), self.input_permutation), loss