# %% [markdown]
# # Deep Imputation of BraTS dataset with MONAI
# 
# The dataset comes from http://medicaldecathlon.com/.  
# Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)  
# Size: 750 4D volumes (484 Training + 266 Testing)  
# Source: BRATS 2016 and 2017 datasets.  
# Challenge: **Drop some of the modalities randomly and reconstruct it by imputing with a 3D U-Net**

# %%
from logger import Logger
logger = Logger(log_level='DEBUG')

# %% [markdown]
# ### Set parameters
# 
# * RUN_ID : set this to prevent overlapped saving of model and data
# 
# * DO_MASK : Set to True if mask is to be applied while training
# * SET_VARIANCE : Set to True if variance is to be trained in loss function
# * PIXEL_DOWNSAMPLE : How much to scale down each axis. In other words, mm per pixel/voxel
# 
# * MAX_EPOCHS
# * VAL_INTERVAL : how frequently the validation code should be run
# * TRAIN_RATIO: proportion of total dataset to be used for training. Rest will be used for validating

# %%
RUN_ID = 20
QR_REGRESSION = True
DO_MASK = True
SET_VARIANCE = True
MAX_EPOCHS = 6000
TRAIN_DATA_SIZE = 50
BATCHSIZE_TRAIN = 2
# PIXEL_DOWNSAMPLE = [2, 2, 2]
VAL_INTERVAL = 10
TRAIN_RATIO = 0.8
RANDOM_SEED = 0
CONTINUE_TRAINING = False
TRAIN_DATA_SHUFFLE = True
ROOT_DIR = "/scratch1/sachinsa/cont_syn"

# test code sanity (for silly errors)
SANITY_CHECK = False
if SANITY_CHECK:
    RUN_ID = 0
    MAX_EPOCHS = 15
    TRAIN_DATA_SIZE = 10
    VAL_INTERVAL = 2

params = {
    'RUN_ID': RUN_ID,
    'QR_REGRESSION': QR_REGRESSION,
    'DO_MASK': DO_MASK,
    'SET_VARIANCE': SET_VARIANCE,
    'MAX_EPOCHS': MAX_EPOCHS,
    'ROOT_DIR': ROOT_DIR
}

logger.info("PARAMETERS\n-----------------")
logger.info(f"RUN_ID: {RUN_ID}")
logger.info(f"QR_REGRESSION: {QR_REGRESSION}")
logger.info(f"DO_MASK: {DO_MASK}")
logger.info(f"SET_VARIANCE: {SET_VARIANCE}")
logger.info(f"MAX_EPOCHS: {MAX_EPOCHS}")
logger.info(f"TRAIN_DATA_SIZE: {TRAIN_DATA_SIZE}")
logger.info(f"BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}")
# logger.info(f"PIXEL_DOWNSAMPLE: {PIXEL_DOWNSAMPLE}")
logger.info(f"VAL_INTERVAL: {VAL_INTERVAL}")
logger.info(f"TRAIN_RATIO: {TRAIN_RATIO}")
logger.info(f"RANDOM_SEED: {RANDOM_SEED}")
logger.info(f"CONTINUE_TRAINING: {CONTINUE_TRAINING}")
logger.info(f"TRAIN_DATA_SHUFFLE: {TRAIN_DATA_SHUFFLE}")
logger.info(f"ROOT_DIR: {ROOT_DIR}")
print("")

# %% [markdown]
# ## Check if this is a notebook or not

# %%
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
    logger.debug("This is a Jupyter Notebook.")
else:
    logger.debug("This is a Python script (not a Jupyter Notebook).")

# %% [markdown]
# ## Setup imports

# %%
import os
import numpy as np
import pdb
import time
import matplotlib.pyplot as plt
import json
import pickle

from monai.config import print_config
from monai.transforms import (
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
    CenterSpatialCrop,
    Spacing,
    EnsureType,
    EnsureChannelFirst,
)
from monai.metrics import MSEMetric
from monai.utils import set_determinism

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BrainMRIDataset

# print_config()

# %%
save_dir = os.path.join(ROOT_DIR, f"run_{RUN_ID}")
if not CONTINUE_TRAINING and os.path.exists(save_dir) and os.path.isdir(save_dir) and len(os.listdir(save_dir)) != 0:
    logger.warning(f"{save_dir} already exists. Avoid overwrite by updating RUN_ID.")
    # exit()
else:
    os.makedirs(save_dir, exist_ok=True)

# %% [markdown]
# ### Set deterministic training for reproducibility

# %%
set_determinism(seed=RANDOM_SEED)

# %% [markdown]
# ## Setup transforms for training and validation
# 

# %%
# crop_size = [224, 224, 144]
# resize_size = [crop_size[i]//PIXEL_DOWNSAMPLE[i] for i in range(len(crop_size))]
resize_size = [64,64,64]

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        # Spacing(
        #     pixdim=(1.0, 1.0, 1.0),
        #     mode=("bilinear", "nearest"),
        # ),
        # RandSpatialCrop(roi_size=crop_size, random_size=False),
        Resize(spatial_size=resize_size, mode='nearest'),
        # RandFlip(prob=0.5, spatial_axis=0),
        # RandFlip(prob=0.5, spatial_axis=1),
        # RandFlip(prob=0.5, spatial_axis=2),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        # RandScaleIntensity(factors=0.1, prob=1.0),
        # RandShiftIntensity(offsets=0.1, prob=1.0),
    ]
)

val_transform = Compose(
    [
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        # Spacing(
        #     pixdim=(1.0, 1.0, 1.0),
        #     mode=("bilinear", "nearest"),
        # ),
        # CenterSpatialCrop(roi_size=crop_size), # added this because model was not handling 155dims
        Resize(spatial_size=resize_size, mode='nearest'),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
)

# %% [markdown]
# ## Load data

# %% [markdown]
# Create training and validation dataset

# %%
from torch.utils.data import Subset

all_dataset = BrainMRIDataset(
    root_dir=ROOT_DIR,
    seed = RANDOM_SEED
)

# Split the dataset
train_dataset, val_dataset = random_split(all_dataset, [TRAIN_RATIO, 1-TRAIN_RATIO],
                                          generator=torch.Generator().manual_seed(RANDOM_SEED))

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

if TRAIN_DATA_SIZE:
    train_dataset = Subset(train_dataset, list(range(TRAIN_DATA_SIZE)))
    val_dataset = Subset(val_dataset, list(range(TRAIN_DATA_SIZE//4)))

BATCHSIZE_VAL = BATCHSIZE_TRAIN

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=TRAIN_DATA_SHUFFLE,
    num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False, num_workers=8)

logger.debug("Data loaded")
logger.debug(f"Length of dataset: {len(train_dataset)}, {len(val_dataset)}")
logger.debug(f"Batch-size: {BATCHSIZE_TRAIN}, {BATCHSIZE_VAL}")
logger.debug(f"Length of data-loaders: {len(train_loader)}, {len(val_loader)}")

# %% [markdown]
# ## Check data shape and visualize

# %%
# pick one image from DecathlonDataset to visualize and check the 4 channels
if is_notebook():
    channels = ["FLAIR", "T1w", "T1gd", "T2w"]
    val_data_example = val_dataset[0]['image']
    _, im_length, im_width, im_height = val_data_example.shape
    logger.debug(f"image shape: {val_data_example.shape}")
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

# %% [markdown]
# ## Create Model, Loss, Optimizer

# %% [markdown]
# **Define a 3D Unet**

# %%
device = torch.device("cuda:0")
out_channels = 12 if QR_REGRESSION else 8
model = UNet(
    spatial_dims=3, # 3D
    in_channels=4,
    out_channels=out_channels,
    channels=(4, 8, 16),
    strides=(2, 2),
    num_res_units=2
).to(device)
logger.debug("Model defined")

# %%
# Calculate and display the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
# logger.debug(f"Total number of trainable parameters: {total_params}")

# Print the model architecture
# logger.debug(f"Model Architecture:\n {model}")

# %%
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
ep_start = 1

epoch_loss_values = []
metric_values = []

if CONTINUE_TRAINING:
    load_dir = save_dir
    checkpoint = torch.load(os.path.join(load_dir, 'latest_checkpoint.pth'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    ep_start = checkpoint['epoch']

    with open(os.path.join(load_dir, 'training_info.pkl'), 'rb') as f:
        training_info = pickle.load(f)
        epoch_loss_values = training_info['epoch_loss_values']
        metric_values = training_info['metric_values']

# %% [markdown]
# ### Define Losses

# %%
def gaussian_nll_loss(outputs, target):
    # input is 4 channel images, outputs is 8 channel images
    outputs_mean = outputs[:, :4, ...]
    if not SET_VARIANCE:
        log_std = torch.zeros_like(outputs_mean) # sigma = 1
    else:
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

def loss_scheduler(epoch):
    if QR_REGRESSION:
        return qr_loss
    else:
        if 500 < epoch < 600:
            return mse_loss
        else:
            return gaussian_nll_loss

# %%
mse_metric = MSEMetric(reduction="mean")

# define inference method
def inference(input):
    def _compute(input):
        output = model(input)
        return output

    with torch.amp.autocast('cuda'):
        return _compute(input)


# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# %%
# Save params
with open(os.path.join(save_dir, 'params.pkl'), 'wb') as f:
    pickle.dump(params, f)

# %%
best_metric = -1
best_metric_epoch = -1

logger.debug("Beginning training...")
total_start = time.time()
for epoch in range(ep_start, MAX_EPOCHS+1):
    epoch_start_time = time.time()
    logger.info("-" * 10)
    logger.info(f"epoch {epoch}/{MAX_EPOCHS}")
    model.train()
    epoch_loss = 0
    step = 0
    criterion = loss_scheduler(epoch)
    step_start = time.time()
    for batch_data in train_loader:
        data_loaded_time = time.time() - step_start
        step += 1
        inputs, mask, id = (
            batch_data["image"].to(device),
            batch_data["mask"].to(device),
            batch_data["id"],
        )
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            target = inputs.clone()
            if DO_MASK:
                inputs = inputs*~mask[:,:,None,None,None]
            # for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"NaN found in gradient of parameter: {name}")
            #         exit()
            outputs = model(inputs)            
            loss = criterion(outputs, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if epoch > 10 and loss.item() > 1e5:
        #     logger.warning(f"large loss encountered: {loss.item()}!")
        #     exit()
        if np.isnan(loss.item()):
            logger.warning("nan value encountered!")
            exit()
        epoch_loss += loss.item()
        logger.info(
            f"{step}/{len(train_loader)}"
            f", train_loss: {loss.item():.4f}"
            f", data-load time: {(data_loaded_time):.4f}"
            f", total-step time: {(time.time() - step_start):.4f}"
        )
        step_start = time.time()
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    logger.info(f"epoch {epoch} average loss: {epoch_loss:.4f}")

    if epoch % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_mask = (
                    batch_data["image"].to(device),
                    batch_data["mask"].to(device),
                )
                val_target = val_inputs.clone()
                val_inputs = val_inputs*~val_mask[:,:,None,None,None]
                val_outputs = inference(val_inputs)
                val_output_main = val_outputs[:,:4,...]
                mse_metric(y_pred=val_output_main, y=val_target)

            metric = 1-mse_metric.aggregate().item()
            metric_values.append(metric)
            mse_metric.reset()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'latest_checkpoint.pth'),
            )
            logger.info(f"saved latest model at epoch: {epoch}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, 'best_checkpoint.pth'),
                )
                logger.info(f"saved new best metric model at epoch: {epoch}")
                
            # Save the loss list
            with open(os.path.join(save_dir, 'training_info.pkl'), 'wb') as f:
                pickle.dump({
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                }, f)
            logger.info(
                f"current epoch: {epoch} current mean mse: {metric:.4f}"
                f" best mean metric: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    logger.info(f"time consuming of epoch {epoch} is: {(time.time() - epoch_start_time):.4f}")
total_time = time.time() - total_start

# %%
logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
logger.info(f"Training time: {total_time//MAX_EPOCHS:.1f}s/ep (total: {total_time//3600:.0f}h {(total_time//60)%60:.0f}m)")


