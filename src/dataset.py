## Custom Dataset to load MRI and mask

import os
import numpy as np
import json
from torch.utils.data import Dataset

def loadBRATS2017(root_dir):
    json_file_path = os.path.join(root_dir, "dataset.json")
    with open(json_file_path, 'r') as file:
        properties = json.load(file)
    return properties

DATA_PATH = {
    'BraTS_2017': '/scratch1/sachinsa/data/Task01_BrainTumour'
}

# TODO: Study MONAI DecathloanDataset and CacheDataset class to improve this class
class BraTSDataset(Dataset):
    def __init__(self, version, section='training', train_ratio=0.8, transform=None, seed=0, has_mask=True, has_label=True):
        self.version = version
        self.root_dir = DATA_PATH[f'BraTS_{version}']
        np.random.seed(seed)

        if version == '2017':
            self.properties = loadBRATS2017(self.root_dir)
            self.ids = np.array([int(filepath['image'][17:-7]) for filepath in self.properties['training']])
            self._prune()
            train_ids, val_ids = self._random_split(self.ids, train_ratio)
            if section == 'training':
                self.ids = train_ids
            elif section == 'validation':
                self.ids = val_ids
        
        # TODO: if transform == None, use a default transform that simply loads the data
        self.transform = transform

    def _random_split(self, _list, split_ratio):
        np.random.shuffle(_list)
        split_idx = int(len(_list) * split_ratio)
        split1 = _list[:split_idx]
        split2 = _list[split_idx:]
        return split1, split2

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        filepath = {
            "image": os.path.normpath(os.path.join(self.root_dir,self.properties['training'][index]['image'])),
            "label": os.path.normpath(os.path.join(self.root_dir,self.properties['training'][index]['label']))
        }
        image_dict = self.transform(filepath)

        return {"id": self.ids[index],
                "image": image_dict["image"],
                "label": image_dict["label"]}
    
    def _prune(self):
        if self.version == '2017':
            # removing BRATS_065 dataset as it disrupts training
            self.ids = self.ids[self.ids != 65]
    
    def get_with_id(self, id):
        idx = self.ids.index(id)
        return self.__getitem__(idx)
    
    def get_ids(self) -> np.ndarray:
        """
        Get the ids of datalist used in this dataset.

        """
        return self.ids
    
    def get_properties(self):
        return self.properties