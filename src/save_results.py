# %% [markdown]
# # To save uncertainty results for downstream task

# %%
RUN_ID = 23
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

from utils.model import create_UNet3D, inference
from utils.transforms import contr_syn_transform
from utils.dataset import BraTSDataset

logger = Logger(log_level='DEBUG')

# %%
load_dir = os.path.join(ROOT_DIR, f"run_{RUN_ID}")
save_dir = os.path.join('/scratch1/sachinsa/data/contr_generated', f"run_{RUN_ID}")
os.makedirs(save_dir, exist_ok=True)

# %%
device = torch.device("cuda:0")
model = create_UNet3D(out_channels=12, device=device)

# %%
all_dataset = BraTSDataset(
    version='2017',
    section='all',
    seed = RANDOM_SEED,
    transform = contr_syn_transform['val']
)
all_loader = DataLoader(all_dataset, batch_size=1, shuffle=False, num_workers=8)

# %%
# Load masks
mask_root_dir = "/scratch1/sachinsa/data/masks/brats2017"
train_mask_df = pd.read_csv(os.path.join(mask_root_dir, "train_mask.csv"), index_col=0)
val_mask_df = pd.read_csv(os.path.join(mask_root_dir, "val_mask.csv"), index_col=0)
all_mask_df = pd.concat([train_mask_df, val_mask_df], axis=0)
all_mask_df.head(2)

# %%
checkpoint = torch.load(os.path.join(load_dir, 'best_checkpoint.pth'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
channels = ["FLAIR", "T1w", "T1Gd", "T2w"]

# %%
import nibabel as nib

# %%
with torch.no_grad():
    for this_data in all_loader:
        this_inputs, this_ids = (
            this_data["image"].to(device),
            this_data["id"],
        )
        this_mask = torch.from_numpy(all_mask_df.loc[this_ids.tolist(), :].values).to(device)
        this_target = this_inputs.clone()
        this_inputs = this_inputs*~this_mask[:,:,None,None,None]
        this_outputs = inference(this_inputs, model)
        
        mri_array = this_outputs[0].detach().permute(1, 2, 3, 0).cpu().numpy()
        nifti_img = nib.Nifti1Image(mri_array,affine=np.eye(4))
        output_filename = os.path.join(save_dir, f'BRATS_{this_ids[0]}.nii.gz')
        print(output_filename)
        nib.save(nifti_img, output_filename)


