import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import torch
import numpy as np

def plot_donut(value, title, color, ax):
    value = max(0, min(1, value))
    ax.axis("equal")

    radius = 0.7
    inner_radius = 0.5
    angle = 360 * value

    background_circle = Wedge((0, 0), radius, 0, 360, width=radius - inner_radius, color="lightgray")
    ax.add_patch(background_circle)

    active_segment = Wedge((0, 0), radius, 0, angle, width=radius - inner_radius, color=color)
    ax.add_patch(active_segment)

    ax.text(0, 0, f"{value:.0%}", ha="center", va="center", fontsize=20, color="black")
    ax.text(0, 0.85, title, ha="center", va="center", fontsize=20, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

def tensor_normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def tensor_clamp(tensor, low=0.01, hi=0.99):
    lower_bound = torch.quantile(tensor, low)
    upper_bound = torch.quantile(tensor, hi)
    return torch.clamp(tensor, min=lower_bound.item(), max=upper_bound.item())

def background_standardize(tensor):
    max_ = tensor.max().item()
    min_ = tensor.min().item()
    tensor[torch.abs(tensor) <  0.03] = min_ # (max_+min_)/2
    return tensor

class BrainPlot():
    def __init__(self, input_image, output_image, input_mask, id, h_index=None, clamp_vis=True):
        self.input_image = input_image
        self.output_image = output_image
        self.input_mask = input_mask
        self.id = id
        im_shape = input_image.shape[2:]
        self.im_length = im_shape[0]
        self.im_width = im_shape[1]
        self.im_height = im_shape[2]
        self.h_index = h_index if h_index is not None else self.im_height//2
        self.clamp_vis = clamp_vis
        self.channels = ["FLAIR", "T1w", "T1Gd", "T2w"]
        self.num_channels = len(self.channels)
        self.total_index = 0

    def plot(self, key):
        index = self.total_index%self.num_channels
        row_title = ""
        this_input_sub = self.input_image[0, :, :, :, self.h_index]
        this_output_sub = self.output_image[0, :, :, :, self.h_index]
        nc = self.num_channels 

        if key == "input":
            row_title = "Input"
            brain_slice = this_input_sub[index]
        elif key == "q0":
            row_title = "Output: \n" + "Median"
            brain_slice = this_output_sub[index]
        elif key == "q1":
            row_title = "Output: \n" + r"qL" + ""
            brain_slice = this_output_sub[index+1*nc]
        elif key == "q2":
            row_title = "Output: \n" + r"qH" + ""
            brain_slice = this_output_sub[index+2*nc]
        elif key == "q3":
            row_title = "Outlier"
            lower_slice = this_input_sub[index] < this_output_sub[index+nc]
            upper_slice = this_input_sub[index] > this_output_sub[index+2*nc]
            brain_slice = torch.logical_or(lower_slice, upper_slice)
        elif key == "diff":
            row_title = "Diff"
            brain_slice = this_input_sub[index]-this_output_sub[index]
        num_rows = 5
        
        plt.subplot(num_rows, 4, 1+self.total_index)
        self.total_index += 1

        col_title = self.channels[index]
        col_color = "red" if self.input_mask[index] else "green"
        if key == "input":
            plt.title(col_title, fontsize=30, color=col_color)
        brain_slice = brain_slice.detach().cpu().T
        brain_slice = torch.flip(brain_slice, dims=[0]) # flip horizontally

        if brain_slice.dtype != torch.bool:
            if self.clamp_vis:
                brain_slice = tensor_clamp(brain_slice)
            brain_slice = background_standardize(brain_slice)

        plt.imshow(brain_slice, cmap="gray")

        plt.xlabel('')
        if index == 0:
            plt.ylabel(row_title, fontsize=30)

        plt.suptitle(f"BRATS_{self.id} (h={self.h_index}/{self.im_height})", fontsize=20)
        plt.xticks([self.im_width - 1], [self.im_width], fontsize=15)
        plt.yticks([self.im_length - 1], [self.im_length], fontsize=15)
        plt.tight_layout()
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.tick_params(labelsize=20)
