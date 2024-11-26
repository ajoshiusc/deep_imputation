import numpy as np
import torch

# Define losses

def gaussian_nll_loss(outputs, target):
    # input is 4 channel images, outputs is 8 channel images
    outputs_mean = outputs[:, :4, ...]
    
    # Set variance to 1
    # log_std = torch.zeros_like(outputs_mean) # sigma = 1
    
    log_std = outputs[:, 4:, ...]
    eps = np.log(1e-6)/2 # -6.9

    # TODO: should the clamping be with or without autograd?
    log_std = log_std.clone()
    with torch.no_grad():
        log_std.clamp_(min=eps)

    cost1 = (target - outputs_mean)**2 / (2*torch.exp(2*log_std))
    cost2 = log_std

    return torch.mean(cost1 + cost2)

def mse_loss(outputs, target):
    return torch.nn.functional.mse_loss(outputs[:, :4, ...], target)

def qr_loss(out, tgt, q0=0.5, q1=0.841, q2=0.159):
    out0 = out[:, :4, ...]
    out1 = out[:, 4:8, ...]
    out2 = out[:, 8:, ...]
    custom_loss0 = torch.sum(torch.max(q0 * (out0 - tgt), (q0 - 1.0) * (out0 - tgt)))
    custom_loss1 = torch.sum(torch.max(q1 * (out1 - tgt), (q1 - 1.0) * (out1 - tgt)))
    custom_loss2 = torch.sum(torch.max(q2 * (out2 - tgt), (q2 - 1.0) * (out2 - tgt)))

    return custom_loss0 + custom_loss1 + custom_loss2