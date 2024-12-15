import torch
import torch.nn as nn
# import wandb

def rearrange_tensor(input_tensor, order):
    """
    Rearrange the input tensor to the desired order
    Args:
        input_tensor: input tensor to be rearranged 
        order: desired order of the tensor  (BTCHW)
    Returns:
        rearranged tensor
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute([order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    """
    Rearrange the input tensor back to the original order
    Args:
        input_tensor: input tensor to be rearranged (BTCHW)
        order: original order of the tensor
    Returns:
        rearranged tensor
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    assert all([dim in "BCHWT" for dim in order]), "Order must not contain any characters other than BCHWT"

    return input_tensor.permute(["BTCHW".index(dim) for dim in order])

class MotionPnLayer(torch.nn.Module):
    def __init__(self, exp_name = "", penalty_weight=1.0):
        super(MotionPnLayer, self).__init__()
        # default configs
        self.input_permutation = "BCTHW"   # Input permutation from video reader
        self.input_color_order = "RGB"     # Input color channel order from video reader
        self.gray_scale = {"B": 0.114, "G": 0.587, "R": 0.299}

        # power normalization parameters
        self.pn = m_sigmoid
        self.m = nn.Parameter(torch.zeros(1) + 1e-1)
        self.n = nn.Parameter(torch.zeros(1))

        # temporal attention variation regularization parameter
        self.lambda1 = penalty_weight
        print("*" * 20, f"Penalty weight: {self.lambda1}")

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
    def forward(self, video_seq):
        # rearrange the input tensor to BTCHW
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        
        # normalize the input tensor back to [0, 1]
        input_std = torch.tensor(self.input_std).view(1, 1, 3, 1, 1).to(video_seq.device).to(video_seq.dtype)
        input_mean = torch.tensor(self.input_mean).view(1, 1, 3, 1, 1).to(video_seq.device).to(video_seq.dtype)
        norm_seq = video_seq * input_std + input_mean
        
        # transfor the input tensor to grayscale 
        weights = torch.tensor([self.gray_scale[idx] for idx in self.input_color_order], 
                               dtype=video_seq.dtype, device=video_seq.device)
        grayscale_video_seq = torch.einsum("btchw, c -> bthw", norm_seq, weights)

        ### frame difference ###
        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:,1:] - grayscale_video_seq[:,:-1]

        ### check if 0 difference, if so duplicate the last non-zero frame difference ###
        # zero_diff = torch.sum(frame_diff == 0.0, dim=(2, 3))
        # for i in range(B):
        #     matching_mask = (zero_diff[i] == H * W)
        #     matching_indices = torch.nonzero(matching_mask).flatten()
        #     for j in matching_indices:
        #         if j > 0:
        #             frame_diff[i, j] = frame_diff[i, j - 1]

        ### power normalization ###
        attention_map = self.pn(frame_diff, self.m, self.n)
        repeat_attention_map = attention_map.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        ### temporal attention variation regularization ###
        loss = 0
        if torch.is_grad_enabled():
            temp_diff = attention_map[:, 1:] - attention_map[:, :-1]
            temporal_loss = torch.sum(temp_diff.pow(2)) / (H*W*(T-2)*B)
            loss = self.lambda1 * temporal_loss
        
        ### element-wise multiplication ###
        prompt = repeat_attention_map * video_seq[:,1:]
        motion_prompt = reverse_rearrange_tensor(prompt, self.input_permutation)

        return motion_prompt, loss


def m_sigmoid(input, m, n):
    return 1 / (1 + torch.exp(
        - (5 / (0.45 * torch.abs(torch.tanh(m)) + 1e-1)) * (input - 0.6 * torch.tanh(n))
        ))