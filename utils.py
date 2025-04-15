# File: utils.py

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np
import os
from PIL import Image

########################################
# 1) Define my_bidict here at the top:
########################################
my_bidict = {
    'Class0': 0,
    'Class1': 1,
    'Class2': 2,
    'Class3': 3
}

def rescaling(x):
    """Rescales a tensor from [0,1] to [-1,1]."""
    return x * 2.0 - 1.0
    
def rescaling_inv(x):
    """Inverse of rescaling: [-1,1] -> [0,1]."""
    return (x + 1.0) / 2.0

def concat_elu(x):
    """
    Like concatenated ReLU (http://arxiv.org/abs/1603.05201),
    but uses ELU for the nonlinearity. 
    Doubling channels from C to 2C by cat(x, -x).
    """
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation."""
    axis = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """Numerically stable log_softmax that prevents overflow."""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def discretized_mix_logistic_loss(x, l):
    """
    Log-likelihood for a mixture of discretized logistics.
    Assumes data is in [-1,1].
    """
    # reorder from [B,C,H,W] to [B,H,W,C]
    x = x.permute(0,2,3,1)
    l = l.permute(0,2,3,1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    nr_mix = ls[-1] // 10  # each mixture has 10 output params for RGB
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # mean, scale, coeff
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2*nr_mix], min=-7.)
    coeffs = F.tanh(l[:, :, :, :, 2*nr_mix:3*nr_mix])

    x = x.contiguous()
    # x shape => [B,H,W,C], we add a mixture dimension => [B,H,W,C, nr_mix]
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix], device=x.device), requires_grad=False)

    # Adjust means according to preceding sub-pixels
    m2 = (means[:, :, :, 1, :] +
          coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] +
          coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0/255.0)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0/255.0)
    cdf_min = F.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability mass in each bin

    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    # robust version
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = (inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) +
                       (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5)))
    inner_cond = (x > 0.999).float()
    inner_out = (inner_cond * log_one_minus_cdf_min +
                 (1.0 - inner_cond) * inner_inner_out)
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))

def to_one_hot(tensor, n, fill_with=1.):
    """One-hot encode a tensor along the last axis."""
    one_hot = torch.zeros(tensor.size() + (n,), device=tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def sample_from_discretized_mix_logistic(l, nr_mix):
    """
    Sampling from a mixture of discretized logistics (RGB).
    Returns [B,3,H,W].
    """
    l = l.permute(0, 2, 3, 1)  # [B,H,W,C]
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]
    
    logit_probs = l[:, :, :, :nr_mix]
    l_rest = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    
    # sample mixture indicator
    temp = torch.rand_like(logit_probs)
    temp = logit_probs - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot_map = to_one_hot(argmax, nr_mix).view(xs[:-1] + [1, nr_mix])

    means = torch.sum(l_rest[:, :, :, :, :nr_mix] * one_hot_map, dim=4)
    log_scales = torch.clamp(
        torch.sum(l_rest[:, :, :, :, nr_mix:2*nr_mix]*one_hot_map, dim=4), min=-7.
    )
    coeffs = torch.sum(
        F.tanh(l_rest[:, :, :, :, 2*nr_mix:3*nr_mix])*one_hot_map, dim=4
    )

    u = torch.rand_like(means)
    x = means + torch.exp(log_scales)*(torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(x[:, :, :, 0], -1.0, 1.0)
    x1 = torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0]*x0, -1.0, 1.0)
    x2 = torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1]*x0 + coeffs[:, :, :, 2]*x1, -1.0, 1.0)

    out = torch.stack([x0, x1, x2], dim=3)
    return out.permute(0, 3, 1, 2)

def down_shift(x, pad=None):
    """
    Shift the image down by removing bottom row and adding a blank row at the top.
    """
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)

def right_shift(x, pad=None):
    """
    Shift the image right by removing right-most column 
    and adding a blank column on the left.
    """
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3] - 1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)

def sample(model, sample_batch_size, obs, sample_op):
    """
    Autoregressive sampling loop for unconditional or conditional PixelCNN.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2], device=device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                out = model(data, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample[:, :, i, j]
    return data

class mean_tracker:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    def update(self, new_value):
        self.sum += new_value
        self.count += 1
    def get_mean(self):
        return (self.sum / self.count) if self.count>0 else 0
    def reset(self):
        self.sum = 0.0
        self.count = 0

class ratio_tracker:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, new_value, new_count):
        self.sum += new_value
        self.count += new_count
    def get_ratio(self):
        return (self.sum / self.count) if self.count>0 else 0
    def reset(self):
        self.sum = 0
        self.count = 0
        
def check_dir_and_create(dir):
    """
    If a directory doesn't exist, create it.
    """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def save_images(tensor, images_folder_path, label=''):
    """
    Save a batch of images (shape [B,3,H,W]) to the given folder, 
    named label_image_XX.png
    """
    os.makedirs(images_folder_path, exist_ok=True)
    for i, img_tensor in enumerate(tensor):
        img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        img = Image.fromarray(img_np, mode='RGB')
        img_path = f"{images_folder_path}/{label}_image_{i+1:02d}.png"
        img.save(img_path)
