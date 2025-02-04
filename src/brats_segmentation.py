# %% [markdown]
# # Brain tumor 3D segmentation with MONAI
# 
# The dataset comes from http://medicaldecathlon.com/.  
# Target: Gliomas segmentation necrotic/active tumour and oedema  
# Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)  
# Size: 750 4D volumes (484 Training + 266 Testing)  
# Source: BRATS 2016 and 2017 datasets.  
# Challenge: Complex and heterogeneously-located targets
# 
# The image patches show from left to right:
# 1. the whole tumor (yellow) visible in T2-FLAIR (Fig.A).
# 1. the tumor core (red) visible in T2 (Fig.B).
# 1. the enhancing tumor structures (light blue) visible in T1Gd, surrounding the cystic/necrotic components of the core (green) (Fig. C).
# 1. The segmentations are combined to generate the final labels of the tumor sub-regions (Fig.D): edema (yellow), non-enhancing solid core (red), necrotic/cystic core (green), enhancing core (blue).

# %%
from utils.logger import Logger
logger = Logger(log_level='DEBUG')

# %%
RUN_ID = 40
MASK_CODE = 0 #RUN_ID - 20
RANDOM_SEED = 0
MAX_EPOCHS = 2000
TRAIN_DATA_SIZE = None
VAL_INTERVAL = 5
BATCHSIZE_TRAIN = 2
ROOT_DIR = "/scratch1/sachinsa/brats_seg"
DATA_ROOT_DIR = "/scratch1/sachinsa/data"

# test code sanity (for silly errors)
SANITY_CHECK = False
if SANITY_CHECK:
    RUN_ID = 0
    MAX_EPOCHS = 2
    TRAIN_DATA_SIZE = 6
    VAL_INTERVAL = 1

logger.info("PARAMETERS\n-----------------")
logger.info(f"RUN_ID: {RUN_ID}")
logger.info(f"MASK_CODE: {MASK_CODE}")
logger.info(f"MAX_EPOCHS: {MAX_EPOCHS}")
logger.info(f"TRAIN_DATA_SIZE: {TRAIN_DATA_SIZE}")
logger.info(f"BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}")
logger.info(f"VAL_INTERVAL: {VAL_INTERVAL}")
logger.info(f"RANDOM_SEED: {RANDOM_SEED}")
logger.info(f"ROOT_DIR: {ROOT_DIR}")
print("")

# %% [markdown]
# ## Setup imports

# %%
import os
import time
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import pickle

from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.utils import set_determinism

import torch
from torch.utils.data import Subset

from utils.dataset import BraTSDataset
from utils.model import create_SegResNet, inference
from utils.transforms import tumor_seg_transform_3 as data_transform

from itertools import chain, combinations

# print_config()

# %%
save_dir = os.path.join(ROOT_DIR, f"run_{RUN_ID}")
if os.path.exists(save_dir) and os.path.isdir(save_dir) and len(os.listdir(save_dir)) != 0:
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
    processed = True,
    seed = RANDOM_SEED,
    transform = data_transform['train']
)

val_dataset = BraTSDataset(
    version='2017',
    section = 'validation',
    processed = True,
    seed = RANDOM_SEED,
    transform = data_transform['val']
)

# TODO: add logic to get subset inside BraTSDataset
if TRAIN_DATA_SIZE:
    train_dataset = Subset(train_dataset, list(range(TRAIN_DATA_SIZE)))
    val_dataset = Subset(val_dataset, list(range(TRAIN_DATA_SIZE//4)))

BATCHSIZE_VAL = BATCHSIZE_TRAIN

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False, num_workers=8)

logger.debug("Data loaded")
logger.debug(f"Length of dataset: {len(train_dataset)}, {len(val_dataset)}")
logger.debug(f"Batch-size: {BATCHSIZE_TRAIN}, {BATCHSIZE_VAL}")
logger.debug(f"Length of data-loaders: {len(train_loader)}, {len(val_loader)}")

# %%
def all_subsets(arr):
    subsets = list(chain.from_iterable(combinations(arr, r) for r in range(0, len(arr))))
    return [list(subset) for subset in subsets]

mask_indices = all_subsets([0, 1, 2, 3])[MASK_CODE]
show_indices = [x for x in [0, 1, 2, 3] if x not in mask_indices]
channels = ["FLAIR", "T1w", "T1Gd", "T2w"]
label_list = ["TC", "WT", "ET"]

logger.info(f"Masked contrasts: {[channels[i] for i in mask_indices]}")

# %% [markdown]
# ## Create Model, Loss, Optimizer

# %% [markdown]
# **Define a SegResNet**

# %%
device = torch.device("cuda:0")
in_channels = len(show_indices)
model = create_SegResNet(in_channels, device)
logger.debug("Model defined")

# %%
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# %% [markdown]
# ### Define Losses

# %%
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

# %%
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

scaler = torch.amp.GradScaler('cuda')
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Train the model

# %%
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

logger.debug("Beginning training...")
total_start = time.time()
for epoch in range(1, MAX_EPOCHS+1):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch}/{MAX_EPOCHS}")
    model.train()
    epoch_loss = 0
    step = 0
    step_start = time.time()
    for train_data in train_loader:
        data_loaded_time = time.time() - step_start
        step += 1
        train_inputs, train_labels, train_ids= (
            train_data["image"].to(device),
            train_data["label"].to(device),
            train_data["id"],
        )
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            train_outputs = model(train_inputs)
            loss = loss_function(train_outputs, train_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    print(f"epoch {epoch} average loss: {epoch_loss:.4f}")

    if epoch % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels, val_ids = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                    val_data["id"],
                )

                val_outputs = inference(val_inputs, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, 'best_checkpoint.pth'),
                )
                logger.info(f"saved new best metric model at epoch: {epoch}")
            with open(os.path.join(save_dir, 'training_info.pkl'), 'wb') as f:
                pickle.dump({
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                    'metric_values_tc': metric_values_tc,
                    'metric_values_wt': metric_values_wt,
                    'metric_values_et': metric_values_et
                }, f)
            print(
                f"current epoch: {epoch} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

# %%
logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
logger.info(f"Training time: {total_time//MAX_EPOCHS:.1f}s/ep (total: {total_time//3600:.0f}h {(total_time//60)%60:.0f}m)")


