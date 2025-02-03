# %% [markdown]
# # Contrast Synthesis with Uncertainty estimation
# ## Using Quantile Regression and U-Nets
# 
# The dataset comes from http://medicaldecathlon.com/.  
# Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)  
# Size: 750 4D volumes (484 Training + 266 Testing)  
# Source: BRATS 2016 and 2017 datasets.  
# Challenge: **Drop some of the modalities randomly and reconstruct it by imputing with a 3D U-Net**

# %%
from utils.logger import Logger
logger = Logger(log_level='DEBUG')

# %% [markdown]
# ### Set parameters
# 
# * RUN_ID : set this to prevent overlapped saving of model and data
# 
# * DO_MASK : Set to True if mask is to be applied while training
# 
# * MAX_EPOCHS
# * VAL_INTERVAL : how frequently the validation code should be run
# * TRAIN_RATIO: proportion of total dataset to be used for training. Rest will be used for validating

# %%
RUN_ID = 32
INPUT_MODALITY = "T1_T2_FLAIR"
QR_REGRESSION = False
DO_MASK = True
MAX_EPOCHS = 6000
TRAIN_DATA_SIZE = 200
BATCHSIZE_TRAIN = 2
VAL_INTERVAL = 10
TRAIN_RATIO = 0.8
RANDOM_SEED = 0
CONTINUE_TRAINING = False
ROOT_DIR = "/scratch1/sachinsa/cont_syn"

# test code sanity (for silly errors)
SANITY_CHECK = False
if SANITY_CHECK:
    RUN_ID = 0
    MAX_EPOCHS = 10
    TRAIN_DATA_SIZE = 10
    VAL_INTERVAL = 2

params = {
    'RUN_ID': RUN_ID,
    'QR_REGRESSION': QR_REGRESSION,
    'DO_MASK': DO_MASK,
    'MAX_EPOCHS': MAX_EPOCHS,
    'ROOT_DIR': ROOT_DIR
}

logger.info("PARAMETERS\n-----------------")
logger.info(f"RUN_ID: {RUN_ID}")
logger.info(f"INPUT_MODALITY: {INPUT_MODALITY}")
logger.info(f"QR_REGRESSION: {QR_REGRESSION}")
logger.info(f"DO_MASK: {DO_MASK}")
logger.info(f"MAX_EPOCHS: {MAX_EPOCHS}")
logger.info(f"TRAIN_DATA_SIZE: {TRAIN_DATA_SIZE}")
logger.info(f"BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}")
logger.info(f"VAL_INTERVAL: {VAL_INTERVAL}")
logger.info(f"TRAIN_RATIO: {TRAIN_RATIO}")
logger.info(f"RANDOM_SEED: {RANDOM_SEED}")
logger.info(f"CONTINUE_TRAINING: {CONTINUE_TRAINING}")
logger.info(f"ROOT_DIR: {ROOT_DIR}")
print("")

# %% [markdown]
# ## Setup imports

# %%
import os
import numpy as np
import pandas as pd
import pdb
import time
import matplotlib.pyplot as plt
import pickle

from monai.config import print_config
from monai.metrics import MSEMetric
from monai.utils import set_determinism

import torch
from torch.utils.data import DataLoader, Subset

from utils.dataset import BraTSDataset
from utils.model import create_UNet3D, inference
from utils.transforms import contr_syn_transform_2  as data_transform

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
# ## Load data

# %% [markdown]
# Create training and validation dataset

# %%
train_dataset = BraTSDataset(
    version='2017',
    section = 'training',
    seed = RANDOM_SEED,
    transform = data_transform['train']
)

val_dataset = BraTSDataset(
    version='2017',
    section = 'validation',
    seed = RANDOM_SEED,
    transform = data_transform['val']
)

# TODO: add logic to get subset inside BraTSDataset
if TRAIN_DATA_SIZE:
    train_dataset = Subset(train_dataset, list(range(TRAIN_DATA_SIZE)))
    val_dataset = Subset(val_dataset, list(range(TRAIN_DATA_SIZE//4)))

BATCHSIZE_VAL = BATCHSIZE_TRAIN

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True,
    num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False, num_workers=8)

logger.debug("Data loaded")
logger.debug(f"Length of dataset: {len(train_dataset)}, {len(val_dataset)}")
logger.debug(f"Batch-size: {BATCHSIZE_TRAIN}, {BATCHSIZE_VAL}")
logger.debug(f"Length of data-loaders: {len(train_loader)}, {len(val_loader)}")

# %%
# Load masks
mask_root_dir = "/scratch1/sachinsa/data/masks/brats2017"
train_mask_df = pd.read_csv(os.path.join(mask_root_dir, "train_mask.csv"), index_col=0)
val_mask_df = pd.read_csv(os.path.join(mask_root_dir, "val_mask.csv"), index_col=0)

# %% [markdown]
# ## Create Model, Loss, Optimizer

# %%
input_filter = []
if INPUT_MODALITY == "ONLY_T1":
    input_filter = [1]
elif INPUT_MODALITY == "T1_T2":
    input_filter = [1,3]
elif INPUT_MODALITY == "T1_T2_FLAIR":
    input_filter = [0,1,3]

# %% [markdown]
# **Define a 3D Unet**

# %%
device = torch.device("cuda:0")
out_channels = 1
model = create_UNet3D(len(input_filter), out_channels, device)
logger.debug("Model defined")

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
from utils.loss import qr_loss, mse_loss, gaussian_nll_loss

def loss_scheduler(epoch):
    if QR_REGRESSION:
        return qr_loss
    else:
        return mse_loss

# %%
mse_metric = MSEMetric(reduction="mean")

scaler = torch.amp.GradScaler('cuda')
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
    for train_data in train_loader:
        data_loaded_time = time.time() - step_start
        step += 1
        train_inputs, train_ids = (
            train_data["image"].to(device),
            train_data["id"],
        )
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            target = train_inputs.clone()[:, [2], ...] # T1Gd
            train_inputs = train_inputs[:,input_filter, ...]
            train_outputs = model(train_inputs)
            loss = criterion(train_outputs, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
                val_inputs, val_ids = (
                    val_data["image"].to(device),
                    val_data["id"],
                )
                val_target = val_inputs.clone()[:, [2], ...] # T1Gd
                val_inputs = val_inputs[:,input_filter, ...]
                val_outputs = inference(val_inputs, model)
                mse_metric(y_pred=val_outputs, y=val_target)

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


