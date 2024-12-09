import torch
import torch.nn as nn
# import wandb

def round_tensor(x, decimals=0):
    scale = 10 ** decimals
    return torch.round(x * scale) / scale


def closest_odd_numbers(num):
    num = round_tensor(num, decimals=3)
    assert num >= 1, "Number must be greater than or equal to 1, num = " + str(num)
    base = torch.floor(num).int().item()

    lower = base if base % 2 != 0 else base - 1
    lower = torch.where(base <= num, lower, lower - 2)
    higher = lower + 2

    higher_weight = (num - lower) / 2
    lower_weight = 1 - higher_weight

    return lower.to(num.device), higher.to(num.device), lower_weight.to(num.device), higher_weight.to(num.device)


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


class MotionLayer(torch.nn.Module):
    def __init__(self, exp_name = "", test=False):
        super(MotionLayer, self).__init__()
        # default configs
        self.input_permutation = "BCTHW"   # Input format from video reader
        self.input_color_order = "RGB"     # Input format from video reader
        self.gray_scale = {"B": 0.114, "G": 0.587, "R": 0.299}
        self.visual = test

        self.h = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(1))
        self.t = nn.Parameter(torch.zeros(1))

        self.m = nn.Parameter(torch.zeros(1)+1e-1)
        self.n = nn.Parameter(torch.zeros(1))

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
    def forward(self, video_seq):
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        loss = 0
        
        # normalize the input tensor back to [0, 1]
        norm_seq = video_seq * torch.tensor(self.input_std).view(1, 1, 3, 1, 1).to(video_seq.device) + torch.tensor(self.input_mean).view(1, 1, 3, 1, 1).to(video_seq.device)
        
        # transfor the input tensor to grayscale 
        weights = torch.tensor([self.gray_scale[idx] for idx in self.input_color_order], 
                               dtype=norm_seq.dtype, device=norm_seq.device)
        grayscale_video_seq = torch.einsum("btcwh, c -> btwh", norm_seq, weights)

        ### frame difference ###
        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:,1:] - grayscale_video_seq[:,:-1]

        ### check if 0 difference, if so duplicate the last non-zero frame difference ###
        zero_diff = torch.sum(frame_diff == 0.0, dim=(2, 3))
        for i in range(B):
            matching_mask = (zero_diff[i] == H * W)
            matching_indices = torch.nonzero(matching_mask).flatten()
            for j in matching_indices:
                if j > 0:
                    frame_diff[i, j] = frame_diff[i, j - 1]

        ### power normalization ###
        norm_attention = attention_map(frame_diff, self.m, self.n).unsqueeze(2)

        ### frame summations / counts ###
        sum_h = torch.sum(frame_diff, dim=2)
        sum_w = torch.sum(frame_diff, dim=3)
        count_h = torch.count_nonzero(frame_diff, dim=2)
        count_w = torch.count_nonzero(frame_diff, dim=3)
        # count_h = torch.sum(frame_diff != 0.5, dim=2)
        # count_w = torch.sum(frame_diff != 0.5, dim=3)
        ratio_h = sum_h / (count_h + 1e-6)
        ratio_w = sum_w / (count_w + 1e-6)

        height_window = reciprocal_auto(self.h, H)
        width_window = reciprocal_auto(self.w, W)
        temporal_window = reciprocal_auto(self.t, T - 1)

        # if self.visual:
        #     wandb.log({
        #         "height_window": height_window.data[0].item(),
        #         "width_window": width_window.data[0].item(),
        #         "temporal_window": temporal_window.data[0].item(),
        #         "m": self.m.data[0].item(),
        #         "n": self.n.data[0].item()
        #     })

        ### spatial smoothing ###
        smoothed_ratio_h = torch.zeros_like(ratio_h, device=ratio_h.device)
        smoothed_ratio_w = torch.zeros_like(ratio_w, device=ratio_w.device)
        for i in range(B):
            smoothed_ratio_h[i] = spatial_smoothing.apply(ratio_h[i].unsqueeze(1), height_window).squeeze(1)
            smoothed_ratio_w[i] = spatial_smoothing.apply(ratio_w[i].unsqueeze(1), width_window).squeeze(1)

        ### outer product (local attention map) ###
        outer_product = torch.einsum("bth, btw -> bthw", smoothed_ratio_w, smoothed_ratio_h)\

        ### temporal smoothing ###
        smoothed_outers = torch.zeros_like(outer_product, device=outer_product.device)
        for i in range(B):
            smoothed_outers[i] = temporal_smoothing.apply(outer_product[i].unsqueeze(1), temporal_window).squeeze(1)

        # ### window max ###
        # norm_outers = torch.zeros_like(smoothed_outers, device=smoothed_outers.device)
        # for j in range(B):
        #     outer_diff = smoothed_outers[j].unsqueeze(1)
        #     padding = torch.cat([outer_diff[0].repeat(int(temporal_window)-1, 1, 1, 1), outer_diff], dim=0)
        #     result = torch.zeros(T-1, 1).to(video_seq.device)
        #     for i in range(T-1):
        #         local_max, _ = torch.max(padding[i:i+int(temporal_window)].view(int(temporal_window), 1, -1), dim=-1)
        #         result[i] = torch.max(local_max, dim=0).values
        #     norm_outers[j] = (outer_diff / (result[:, :, None, None] + 1e-6)).squeeze(1)
            
        ### power normalization ###
        norm_attention = attention_map(smoothed_outers, self.m, self.n).unsqueeze(2)
        pad_norm_attention = norm_attention.repeat(1, 1, 3, 1, 1)

        # if torch.is_grad_enabled():
        #     temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
        #     temporal_loss = torch.sum(temp_diff.pow(2)) / (H*W*(T-2)*B)
        #     loss = self.lambda1 * temporal_loss
        #     if self.visual:
        #         wandb.log({
        #             "temporal_loss": loss
        #         })

        # if self.visual:
        #     wandb.log({
        #     "self.a.mean": self.a.data[0].mean(), "self.a.std": self.a.data[0].std(), 
        #     "self.b.mean": self.b.data[0].mean(), "self.b.std": self.b.data[0].std()
        #     })

        return reverse_rearrange_tensor((pad_norm_attention * video_seq[:,1:]), self.input_permutation), loss


# sigmoid inverse function
def reciprocal_auto(param, bound, slope=100):
    """
    apply sigmoid function to map the parameter to the range of [1, bound]

    Args:
    - param: the input parameter
    - bound: the upper bound of the output
    - slope: the slope of the sigmoid function
    """
    slope = 10 if bound <= 32 else slope

    window_mapping = (1 - bound) / (1 + torch.exp(-slope * param)) + bound
    return window_mapping


class spatial_smoothing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, window_size):
        assert window_size <= input.shape[-1], f"Window size must be less than or equal to the input size: {window_size} vs {input.shape[-1]}"
        ctx.window_size = window_size
        window_low, window_high, window_low_weight, window_high_weight = closest_odd_numbers(window_size)

        # window size convolution for low window size
        pad_low = (window_low - 1).div(2, rounding_mode='floor').int().item()
        padded_low = torch.nn.functional.pad(input, (pad_low, pad_low), mode='replicate')
        window_low_k = torch.ones((1, 1, window_low), device=input.device, dtype=input.dtype)
        conv_low = torch.nn.functional.conv1d(padded_low, window_low_k) 

        # window size convolution for high window size
        if window_low_weight != 1:
            pad_high = (window_high - 1).div(2, rounding_mode='floor').int().item()
            padded_high = torch.nn.functional.pad(input, (pad_high, pad_high), mode='replicate')
            window_high_k = torch.ones((1, 1, window_high), device=input.device, dtype=input.dtype)
            conv_high = torch.nn.functional.conv1d(padded_high, window_high_k)
            
            ctx.save_for_backward(conv_low, conv_high)
            return window_low_weight * conv_low  / window_low + window_high_weight * conv_high / window_high
        else:
            ctx.save_for_backward(conv_low) 
            return conv_low / window_low

    @staticmethod
    def backward(ctx, grad_output):
        """
        d_grad_output/d_input = conv1d(grad_output, window) --> Not needed
        d_grad_output/d_window = grad_output * conv1d(input, window) * (-1/window^2)
        """
        grad_window = None
        window_low, window_high, window_low_weight, window_high_weight = closest_odd_numbers(ctx.window_size)
        
        if window_low_weight != 1:
            conv_low, conv_high = ctx.saved_tensors
        else:
            conv_low, = ctx.saved_tensors
        
        if ctx.needs_input_grad[1]:
            if window_low_weight != 1:
                grad_window_low = grad_output * conv_low / - (window_low ** 2)
                grad_window_high = grad_output * conv_high / - (window_high ** 2)
                grad_window = window_low_weight * grad_window_low + window_high_weight * grad_window_high 
            else:
                grad_window = grad_output * conv_low / - (window_low ** 2)

        return None, grad_window
    

class temporal_smoothing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, window_size):
        ctx.window_size = window_size
        window_low, window_high, window_low_weight, window_high_weight = closest_odd_numbers(window_size)
        assert window_high <= input.shape[0]+1 or window_high_weight < 1e-6, f"Window size must be less than or equal to the input size: {window_high} with weight {window_high_weight} vs {input.shape[0]}"

        # zero padding for the input
        pad_low = (window_low - 1).div(2, rounding_mode='floor').int().item()
        permute_input = input.permute(1, 0, 2, 3).unsqueeze(1)
        padded_low = torch.nn.functional.pad(permute_input, (0, 0, 0, 0, pad_low, pad_low))
        window_low_k = torch.ones((1, 1, window_low, 1, 1), device=input.device, dtype=input.dtype)
        conv_low = torch.nn.functional.conv3d(padded_low, window_low_k).squeeze(1).permute(1, 0, 2, 3)

        if window_low_weight != 1:
            pad_high = (window_high - 1).div(2, rounding_mode='floor').int().item()
            padded_high = torch.nn.functional.pad(permute_input, (0, 0, 0, 0, pad_high, pad_high))
            window_high_k = torch.ones((1, 1, window_high, 1, 1), device=input.device, dtype=input.dtype)
            conv_high = torch.nn.functional.conv3d(padded_high, window_high_k).squeeze(1).permute(1, 0, 2, 3)
            ctx.save_for_backward(conv_low, conv_high)
            return window_low_weight * conv_low / window_low + window_high_weight * conv_high / window_high
        else:
            ctx.save_for_backward(conv_low)
            return conv_low / window_low

    @staticmethod
    def backward(ctx, grad_output):
        window_low, window_high, window_low_weight, window_high_weight = closest_odd_numbers(ctx.window_size)
        device = grad_output.device
        
        grad_input, grad_window = None, None

        if ctx.needs_input_grad[0]:
            pad_low = (window_low - 1).div(2, rounding_mode='floor').int().item()
            permute_grad_output = grad_output.permute(1, 0, 2, 3).unsqueeze(1)
            window_low_k = torch.ones((1, 1, window_low, 1, 1), device=device, dtype=grad_output.dtype) / window_low
            grad_input = torch.nn.functional.conv_transpose3d(permute_grad_output, window_low_k, padding=(pad_low,0,0)).squeeze(1).permute(1, 0, 2, 3) * window_low_weight

            if window_high_weight != 0:
                pad_high = (window_high - 1).div(2, rounding_mode='floor').int().item()
                window_high_k = torch.ones((1, 1, window_high, 1, 1), device=device, dtype=grad_output.dtype) / window_high
                grad_input += torch.nn.functional.conv_transpose3d(permute_grad_output, window_high_k, padding=(pad_high,0,0)).squeeze(1).permute(1, 0, 2, 3) * window_high_weight

        if window_high_weight > 0:
            conv_low, conv_high = ctx.saved_tensors
        else:
            conv_low, = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_window = (grad_output * conv_low / - (window_low ** 2)) * window_low_weight
            if window_high_weight > 0:
                grad_window_high = grad_output * conv_high / - (window_high ** 2)
                grad_window += window_high_weight * grad_window_high
                
        return grad_input, grad_window
    

def attention_map(input, m, n):
    return 1 / (1 + torch.exp(
        - (5 / (0.45 * torch.abs(torch.tanh(m)) + 1e-1)) * (input - 0.6 * torch.tanh(n))
        ))