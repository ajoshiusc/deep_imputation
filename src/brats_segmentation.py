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
from logger import Logger
logger = Logger(log_level='DEBUG')

# %%
RUN_ID = 0
MAX_EPOCHS = 2000
TRAIN_DATA_SIZE = None
BATCHSIZE_TRAIN = 2
VAL_INTERVAL = 10
# TRAIN_RATIO = 0.8
RANDOM_SEED = 0
ROOT_DIR = "/scratch1/sachinsa/brats_seg"
DATA_ROOT_DIR = "/scratch1/sachinsa/data"

# test code sanity (for silly errors)
SANITY_CHECK = False
if SANITY_CHECK:
    RUN_ID = 0
    MAX_EPOCHS = 15
    TRAIN_DATA_SIZE = 10
    VAL_INTERVAL = 2

logger.info("PARAMETERS\n-----------------")
logger.info(f"RUN_ID: {RUN_ID}")
logger.info(f"MAX_EPOCHS: {MAX_EPOCHS}")
logger.info(f"TRAIN_DATA_SIZE: {TRAIN_DATA_SIZE}")
logger.info(f"BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}")
logger.info(f"VAL_INTERVAL: {VAL_INTERVAL}")
# logger.info(f"TRAIN_RATIO: {TRAIN_RATIO}")
logger.info(f"RANDOM_SEED: {RANDOM_SEED}")
logger.info(f"ROOT_DIR: {ROOT_DIR}")
print("")

# %% [markdown]
# ## Setup imports

# %%
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import pickle

from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
# import onnxruntime
from tqdm import tqdm

import torch

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
# ### Define a new transform to convert brain tumor labels
# 
# Here we convert the multi-classes labels into multi-labels segmentation task in One-Hot format.

# %%
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

# %% [markdown]
# ## Setup transforms for training and validation

# %%
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# %% [markdown]
# ## Load data

# %% [markdown]
# Create training and validation dataset

# %% [markdown]
# ## Quickly load data with DecathlonDataset
# 
# Here we use `DecathlonDataset` to automatically download and extract the dataset.
# It inherits MONAI `CacheDataset`, if you want to use less memory, you can set `cache_num=N` to cache N items for training and use the default args to cache all the items for validation, it depends on your memory size.

# %%
from torch.utils.data import Subset

# here we don't cache any data in case out of memory issue
train_dataset = DecathlonDataset(
    root_dir=DATA_ROOT_DIR,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=True,
    cache_rate=0.0,
    num_workers=8,#4,
)
val_dataset = DecathlonDataset(
    root_dir=DATA_ROOT_DIR,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=8,#4,
)

if TRAIN_DATA_SIZE:
    train_dataset = Subset(train_dataset, list(range(TRAIN_DATA_SIZE)))
    val_dataset = Subset(val_dataset, list(range(TRAIN_DATA_SIZE//4)))

BATCHSIZE_VAL = BATCHSIZE_TRAIN
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False, num_workers=8)

logger.debug("Data loaded")
logger.debug(f"Length of dataset: {len(train_dataset)}, {len(val_dataset)}")
logger.debug(f"Batch-size: {BATCHSIZE_TRAIN}, {BATCHSIZE_VAL}")
logger.debug(f"Length of data-loaders: {len(train_loader)}, {len(val_loader)}")

# %% [markdown]
# ## Create Model, Loss, Optimizer

# %% [markdown]
# **Define a SegResNet**

# %%
device = torch.device("cuda:0")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)
logger.debug("Model defined")

# %%
# Calculate and display the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
logger.debug(f"Total number of trainable parameters: {total_params}")

# Print the model architecture
# logger.debug(f"Model Architecture:\n {model}")

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


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.amp.autocast('cuda'):
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Execute a typical PyTorch training process

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
    for batch_data in train_loader:
        data_loaded_time = time.time() - step_start
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
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
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
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
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

# %%
logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
logger.info(f"Training time: {total_time//MAX_EPOCHS:.1f}s/ep (total: {total_time//3600:.0f}h {(total_time//60)%60:.0f}m)")


