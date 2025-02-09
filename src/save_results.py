# %% [markdown]
# # To save synthetic results for downstream task

# %%
RUN_ID = 401
RANDOM_SEED = 0
ROOT_DIR = "/scratch1/sachinsa/cont_syn"

# %%
import os
import pandas as pd

import pdb
import numpy as np
from utils.logger import Logger

import torch
from torch.utils.data import DataLoader
import nibabel as nib

from utils.model import create_UNet3D, inference
from utils.transforms import contr_syn_transform_2 as data_transform
from utils.dataset import BraTSDataset

logger = Logger(log_level='DEBUG')

# %%
save_dir = os.path.join('/scratch1/sachinsa/data/contr_generated', f"run_{RUN_ID:03d}")
os.makedirs(save_dir, exist_ok=True)

# %%
all_dataset = BraTSDataset(
    version='2017',
    section='all',
    seed = RANDOM_SEED,
    transform = data_transform['val']
)
all_loader = DataLoader(all_dataset, batch_size=1, shuffle=False, num_workers=8)

# %%
# temp code: to save blank t1gd
device = torch.device("cuda:0")

with torch.no_grad():
    for this_data in all_loader:
        this_inputs, this_ids = (
            this_data["image"].to(device),
            this_data["id"],
        )
        this_outputs = torch.cat([this_inputs[:, :2, ...], torch.zeros_like(this_inputs[:,2:3,...]), this_inputs[:, 3:, ...]], dim=1)
        
        mri_array = this_outputs[0].detach().permute(1, 2, 3, 0).cpu().numpy()
        nifti_img = nib.Nifti1Image(mri_array,affine=np.eye(4))
        output_filename = os.path.join(save_dir, f'BRATS_{this_ids[0]:03d}.nii.gz')
        print(output_filename)
        nib.save(nifti_img, output_filename)

# %%
raise KeyboardInterrupt

# %%
input_filter = [0,1,3]

# %%
device = torch.device("cuda:0")
out_channels = 1
model = create_UNet3D(len(input_filter), out_channels, device)

# %%
load_dir = os.path.join(ROOT_DIR, f"run_{RUN_ID:03d}")
checkpoint = torch.load(os.path.join(load_dir, 'best_checkpoint.pth'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
channels = ["FLAIR", "T1w", "T1Gd", "T2w"]

# %%
with torch.no_grad():
    for this_data in all_loader:
        this_inputs, this_ids = (
            this_data["image"].to(device),
            this_data["id"],
        )
        this_inputs = this_inputs[:,input_filter, ...]
        this_outputs = inference(this_inputs, model)
        this_combined = torch.cat([this_inputs[:, :2, ...], this_outputs, this_inputs[:, 2:, ...]], dim=1)
        
        mri_array = this_combined[0].detach().permute(1, 2, 3, 0).cpu().numpy()
        nifti_img = nib.Nifti1Image(mri_array,affine=np.eye(4))
        output_filename = os.path.join(save_dir, f'BRATS_{this_ids[0]:03d}.nii.gz')
        print(output_filename)
        nib.save(nifti_img, output_filename)


