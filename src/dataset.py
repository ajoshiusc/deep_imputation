## Custom Dataset to load MRI and mask

import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, random_split
import torch

DATA_PATH = {
    'BraTS_2017': '/scratch1/sachinsa/data/Task01_BrainTumour'
}

class Mask():
    def __init__(self, num_samples, num_seq = 4, seed=0):
        np.random.seed(seed)
        mask_drop_code = np.random.randint(0, 2**(num_seq) - 1, size=num_samples)
        self.mask = self.int_to_bool_binary(mask_drop_code, length=num_seq)
        pass

    def int_to_bool_binary(self, int_list, length):
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
    
    def get(self, index):
        return self.mask[index]


class BraTSDataset(Dataset):
    def __init__(self, version, transform=None, seed=0, has_mask=True, has_label=True):
        self.root_dir = DATA_PATH[f'BraTS_{version}']

        if version == '2017':
            json_file_path = os.path.join(self.root_dir, "dataset.json")
            with open(json_file_path, 'r') as file:
                data_json = json.load(file)
            self.image_filenames = data_json['training']
            # removing BRATS_065 dataset as it disrupts training
            self.image_filenames = [item for item in self.image_filenames if 'BRATS_065' not in item['image']]
        
        self.has_mask = has_mask
        self.mask = Mask(num_samples=len(self.image_filenames), seed=seed)
        self.ids = [filepath['image'][11:-7] for filepath in self.image_filenames]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.normpath(os.path.join(self.root_dir,self.image_filenames[idx]['image']))
        mask = self.mask.get(idx)
        image = self.transform(img_name)
        mask = torch.from_numpy(mask)

        return {"id": self.image_filenames[idx]['image'][11:20], "image":image, "mask":mask}
    
    def get_with_id(self, id):
        idx = self.ids.index(id)
        return self.__getitem__(idx)