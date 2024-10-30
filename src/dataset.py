## Custom Dataset to load MRI and mask

import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, random_split
import torch

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


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, seed=0, transform=None):
        self.root_dir = os.path.join(root_dir, "Task01_BrainTumour")
        json_file_path = os.path.join(self.root_dir, "dataset.json")
        with open(json_file_path, 'r') as file:
            data_json = json.load(file)

        self.image_filenames = data_json['training']
        
        # removing BRATS_065 dataset as it disrupts training
        self.image_filenames = [item for item in self.image_filenames if 'BRATS_065' not in item['image']]

        np.random.seed(seed)
        num_seq = 4
        mask_drop_code = np.random.randint(0, 2**(num_seq) - 1, size=len(self.image_filenames))
        self.seq_mask = int_to_bool_binary(mask_drop_code, length=num_seq)

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.normpath(os.path.join(self.root_dir,self.image_filenames[idx]['image']))
        mask = self.seq_mask[idx]
        
        if self.transform:
            image = self.transform(img_name)

        mask = torch.from_numpy(mask)

        return {"id": self.image_filenames[idx]['image'][11:20], "image":image, "mask":mask}