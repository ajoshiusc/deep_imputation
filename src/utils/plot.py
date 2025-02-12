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

    ax.text(0, 0, f"{value:.1%}", ha="center", va="center", fontsize=20, color="black")
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

def plot_brainmri(img, channels = ["FLAIR", "T1w", "T1Gd", "T2w"], h_index=77, horiz=True, no_batch=False):
    if no_batch:
        img = img.unsqueeze(dim=0)
    num_channels = len(channels)
    if horiz:
        fig, axes = plt.subplots(1, num_channels, figsize=(2.2*num_channels, 2), constrained_layout=True)
    else:
        fig, axes = plt.subplots(num_channels, 1, figsize=(2.2, 2*num_channels), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i in range(num_channels):
        axes[i].set_title(channels[i])
        brain_img = img[0, i, ...].detach().cpu()
        print(f"{channels[i]}:\t{brain_img.mean().item():.3f} ± {brain_img.std().item():.3f} [{brain_img.min().item():.3f}, {brain_img.max().item():.3f}]")
        brain_slice = brain_img[..., h_index].T
        im = axes[i].imshow(brain_slice, cmap="gray", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()

def plot_label(img, h_index=77):
    label_list = ["TC", "WT", "ET", "Combined"]
    start_index = 0
    title = "Tumor masks"
    plt.figure(figsize=(12, 8))

    for label_index in range(len(label_list)):
        brain_slice = img[..., h_index].detach().cpu()
        if label_index < 3:
            brain_slice = brain_slice[label_index, ...]
        else:
            brain_slice = brain_slice.sum(axis=0)
        brain_slice = brain_slice.T
        plt.subplot(3, 4, start_index + label_index + 1)
        plt.title(label_list[label_index], fontsize=15)
        if label_index == 0:
            plt.ylabel(title, fontsize=15)
        plt.xticks([])
        plt.yticks([])
        cmap = "gray" if label_index < 3 else "magma"
        plt.imshow(brain_slice, cmap=cmap)

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

def plot_training_synth(epoch_loss_values, metric_values):
    MULTI_TRAINING_FIGURE = True
    val_interval = len(epoch_loss_values)//len(metric_values)

    if not MULTI_TRAINING_FIGURE:
        plt.figure("train", (6, 4))
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss - log")
        plt.yscale('log')
        plt.plot(x, y, color="red")
        plt.title("Training: Gaussian Log Likelihood Loss", fontsize=25)
        plt.savefig(os.path.join(fig_save_dir, "train_plot.png"), facecolor='white')
        plt.show()
    else:
        plt.figure("train", (18, 4))
        plt.subplot(1, 3, 1)
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch", fontsize=15)
        plt.ylabel("loss", fontsize=15)
        plt.plot(x, y, color="red")
        plt.suptitle("Training: Loss", fontsize=20)

        k = 2
        for zoom in [20, 100]:
            if len(x) > zoom:
                plt.subplot(1, 3, k)
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.ylabel("loss", fontsize=15)
                plt.xlabel(f"epoch (from ep. {zoom})", fontsize=15)
                
                plt.plot(x[zoom:], y[zoom:], color="red")
            k += 1
        # plt.savefig(os.path.join(fig_save_dir, "train_plot.png"), facecolor='white')
        plt.show()

    plt.figure("val", (6, 4))
    plt.title("Validation: 1-MSE", fontsize=20)
    x_val = [val_interval * (i + 1) for i in range(len(metric_values))]
    y_val = metric_values
    plt.xlabel("epoch", fontsize=15)
    plt.tight_layout()
    plt.plot(x_val, y_val, color="green")
    # plt.savefig(os.path.join(fig_save_dir, "val_plot.png"), facecolor='white')
    plt.show()


def plot_training_tumor_seg(epoch_loss_values, metric_values, metric_values_tc, metric_values_wt, metric_values_et, val_interval):
    plt.figure("train", (12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.ylim((0, 1))
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.ylim((0, 1))
    plt.plot(x, y, color="green")
    plt.show()

    plt.figure("train", (18, 4))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
    y = metric_values_tc
    plt.xlabel("epoch")
    plt.ylim((0, 1))
    plt.plot(x, y, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
    y = metric_values_wt
    plt.xlabel("epoch")
    plt.ylim((0, 1))
    plt.plot(x, y, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
    y = metric_values_et
    plt.xlabel("epoch")
    plt.ylim((0, 1))
    plt.plot(x, y, color="purple")
    plt.show()

def find_ratio_of_ones(tensor):
    total_elements = tensor.numel()
    count_ones = torch.sum(tensor)
    ratio = count_ones.float() / total_elements
    return ratio.item()

def tensor_to_sorted_tuple(tensor):
    unique_values, counts = torch.unique(tensor, return_counts=True)
    result = list(zip(unique_values.tolist(), counts.tolist()))
    result.sort(key=lambda x: x[1], reverse=True)
    return tuple(result)
    
def find_centroid_3d(tensor):
    print(f"TC: {find_ratio_of_ones(tensor)*100:.5f}%")
    indices = torch.nonzero(tensor, as_tuple=False)

    # print(len(indices))
    # print(tensor_to_sorted_tuple(indices[:, 2]))

    if indices.shape[0] == 0:
        shape = torch.tensor(tensor.shape, dtype=torch.float32)
        midpoint = (shape - 1) / 2
        return midpoint.round().to(torch.int64)

    centroid = torch.mean(indices.float(), dim=0).round().to(torch.int64)
    return centroid