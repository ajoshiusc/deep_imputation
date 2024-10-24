#!/usr/bin/env python
# coding: utf-8

# # Deep Imputation of BraTS dataset with MONAI
# 
# The dataset comes from http://medicaldecathlon.com/.  
# Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)  
# Size: 750 4D volumes (484 Training + 266 Testing)  
# Source: BRATS 2016 and 2017 datasets.  
# Challenge: **Drop some of the modalities randomly and reconstruct it by imputing with a 3D U-Net**

# ### Set parameters
# 
# * run_id : set this to prevent overlapped saving of model and data
# 
# * DO_MASK : Set to True if mask is to be applied while training
# * SET_VARIANCE : Set to True if variance is to be trained in loss function
# * PIXEL_DOWNSAMPLE : How much to scale down each axis. In other words, mm per pixel/voxel
# 
# * max_epochs
# * val_interval : how frequently the validation code should be run

# In[ ]:


run_id = 8
DO_MASK = False
SET_VARIANCE = False
PIXEL_DOWNSAMPLE = [4, 4, 4]
max_epochs = 2000
val_interval = 10
RANDOM_SEED = 0

print("run_id: ", run_id)
print("DO_MASK: ", DO_MASK)
print("SET_VARIANCE: ", SET_VARIANCE)
print("PIXEL_DOWNSAMPLE: ", PIXEL_DOWNSAMPLE)
print("max_epochs: ", max_epochs)
print("val_interval: ", val_interval)
print("RANDOM_SEED: ", RANDOM_SEED)


# ## Check if this is a notebook or not

# In[ ]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or Jupyter QtConsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types
    except NameError:
        return False  # Standard Python interpreter

# Example usage
if is_notebook():
    print("This is a Jupyter Notebook.")
else:
    print("This is a Python script (not a Jupyter Notebook).")


# ## Setup environment

# In[ ]:


if is_notebook():
    get_ipython().system('python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"')
    get_ipython().system('python -c "import matplotlib" || pip install -q matplotlib')
    get_ipython().run_line_magic('matplotlib', 'inline')


# ## Setup imports

# In[ ]:


import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImage,
    Resize,
    NormalizeIntensity,
    Orientation,
    RandFlip,
    RandScaleIntensity,
    RandShiftIntensity,
    RandSpatialCrop,
    Spacing,
    EnsureType,
    EnsureChannelFirst,
)
from monai.metrics import MSEMetric
from monai.utils import set_determinism
from tqdm import tqdm

import pdb
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json

print_config()


# ## Setup data directory
# 
# You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.  
# This allows you to save results and reuse downloads.  
# If not specified a temporary directory will be used.

# In[ ]:


os.environ["MONAI_DATA_DIRECTORY"] = "/scratch1/sachinsa/monai_data_1"


# In[ ]:


directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# In[ ]:


save_dir = os.path.join(root_dir, f"run_{run_id}")
if os.path.exists(save_dir) and os.path.isdir(save_dir) and len(os.listdir(save_dir)) != 0:
    print(f"{save_dir} already exists. Avoid overwrite by updating run_id.")
    exit()
else:
    os.makedirs(save_dir, exist_ok=True)


# ## Set deterministic training for reproducibility

# In[ ]:


set_determinism(seed=RANDOM_SEED)


# ## Setup transforms for training and validation
# 

# In[ ]:


crop_size = [224, 224, 144]
resize_size = [crop_size[i]//PIXEL_DOWNSAMPLE[i] for i in range(len(crop_size))]

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCrop(roi_size=crop_size, random_size=False),
        Resize(spatial_size=resize_size),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        RandScaleIntensity(factors=0.1, prob=1.0),
        RandShiftIntensity(offsets=0.1, prob=1.0),
    ]
)

val_transform = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Resize(spatial_size=resize_size),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
)


# ## Custom Dataset to load MRI and mask

# In[ ]:


def int_to_bool_binary(int_list, length):
    # Convert each integer to its base-2 value and represent it as boolean, always ensuring length is 4
    bool_list = []
    
    for num in int_list:
        # Get the binary representation of the integer (excluding the '0b' prefix)
        binary_str = bin(num)[2:]
        # Convert each character in the binary string to a boolean
        bools = [char == '1' for char in binary_str]
        # Prepend False (0s) to make the length exactly 4
        bools_padded = [False] * (length - len(bools)) + bools
        bool_list.append(bools_padded)
    
    return np.array(bool_list)


# In[ ]:


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.join(root_dir, "Task01_BrainTumour")
        json_file_path = os.path.join(self.root_dir, "dataset.json")
        with open(json_file_path, 'r') as file:
            data_json = json.load(file)

        self.image_filenames = data_json['training']

        np.random.seed(RANDOM_SEED)
        num_seq = 4
        if DO_MASK:
            mask_drop_code = np.random.randint(0, 2**(num_seq) - 1, size=len(self.image_filenames))
            self.seq_mask = int_to_bool_binary(mask_drop_code, length=num_seq)
        else:
            self.seq_mask = np.full((len(self.image_filenames), num_seq), False, dtype=bool)

        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.normpath(os.path.join(self.root_dir,self.image_filenames[idx]['image']))
        mask = self.seq_mask[idx]
        
        if self.transform:
            image = self.transform(img_name)

        mask = torch.from_numpy(mask)

        return {"image":image, "mask":mask}


# Create training and validation dataset

# In[ ]:


all_dataset = BrainMRIDataset(
    root_dir=root_dir
)

# Define the split sizes
train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size
split_sizes = [train_size, val_size]

# Split the dataset
train_dataset, val_dataset = random_split(all_dataset, split_sizes)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(len(train_loader), len(val_loader))


# In[ ]:


# temp code
print("train")
print(len(train_dataset))
print(train_loader.batch_size)
print(len(train_dataset) // train_loader.batch_size)
print("val")
print(len(val_dataset))
print(val_loader.batch_size)


# ## Check data shape and visualize

# In[ ]:


# pick one image from DecathlonDataset to visualize and check the 4 channels
if is_notebook():
    channels = ["FLAIR", "T1w", "T1gd", "T2w"]
    val_data_example = val_dataset[6]['image']
    _, im_length, im_width, im_height = val_data_example.shape
    print(f"image shape: {val_data_example.shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(channels[i], fontsize=30)
        brain_slice = val_data_example[i, :, :, im_height//2].detach().cpu().T
        plt.xticks([0, im_width - 1], [0, im_width - 1], fontsize=15)
        plt.yticks([0, im_length - 1], [0, im_length - 1], fontsize=15)
        plt.imshow(brain_slice, cmap="gray")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
    plt.show()


# ## Create Model, Loss, Optimizer

# **Define a 3D Unet**

# In[ ]:


device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3, # 3D
    in_channels=4,
    out_channels=8, # we will output estimated mean and estimated std dev for all 4 image channels
    channels=(4, 8, 16),
    strides=(2, 2),
    num_res_units=2
).to(device)


# In[ ]:


# Calculate and display the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")

# Print the model architecture
print("Model Architecture:\n", model)


# In[ ]:


def GaussianLikelihood(expected_img, output_img):
    # input is 4 channel images, output is 8 channel images

    output_img_mean = output_img[:, :4, ...]
    if SET_VARIANCE:
        output_img_log_std = output_img[:, 4:, ...]
    else:
        output_img_log_std = torch.zeros_like(output_img[:, 4:, ...]) # sigma = 1

    cost1 = (expected_img - output_img_mean)**2 / (2*torch.exp(2*output_img_log_std))
    cost2 = output_img_log_std

    return torch.mean(cost1 + cost2)


# In[ ]:


VAL_AMP = True

# Define the loss function
loss_function = GaussianLikelihood #nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

mse_metric = MSEMetric(reduction="mean")
mse_metric_batch = MSEMetric(reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        output = model(input)
        return output

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


# In[ ]:


best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, mask = (
            batch_data["image"].to(device),
            batch_data["mask"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs_gt = inputs.clone()
            inputs = inputs*~mask[:,:,None,None,None]
            outputs = model(inputs)
            loss = loss_function(outputs_gt, outputs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_loader)}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_mask = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                val_outputs_gt = val_inputs.clone()
                val_inputs = val_inputs*~val_mask[:,:,None,None,None]
                val_outputs = inference(val_inputs)
                val_outputs = val_outputs[:,:4,...]
                # val_outputs = [post_trans(i) for i in val_outputs]
                mse_metric(y_pred=val_outputs, y=val_outputs_gt)
                mse_metric_batch(y_pred=val_outputs, y=val_outputs_gt)

            metric = mse_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = mse_metric_batch.aggregate()
            mse_metric.reset()
            mse_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean mse: {metric:.4f}"
                f"\nbest mean metric: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start


# In[ ]:


print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


# In[ ]:


# Save the loss list
np.save(os.path.join(save_dir, 'epoch_loss_values.npy'), np.array(epoch_loss_values))
np.save(os.path.join(save_dir, 'metric_values.npy'), np.array(metric_values))
del epoch_loss_values, metric_values


# ## Cleanup data directory
# 
# Remove directory if a temporary was used.

# In[ ]:


if directory is None:
    shutil.rmtree(root_dir)

