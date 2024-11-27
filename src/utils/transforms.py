from monai.transforms import (
    Compose,
    # Invertd,
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
import torch

contr_syn_transform_1 = {
    'train': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCrop(roi_size=[224, 224, 144], random_size=False),
        Resize(spatial_size=[64,64,64], mode='nearest'),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        RandScaleIntensity(factors=0.1, prob=1.0),
        RandShiftIntensity(offsets=0.1, prob=1.0),
    ]),
    'val': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CenterSpatialCrop(roi_size=[224, 224, 144]), # added this because model was not handling 155dims
        Resize(spatial_size=[64,64,64], mode='nearest'),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]),
    'test': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CenterSpatialCrop(roi_size=[224, 224, 144]),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
}

contr_syn_transform_2 = {
    'train': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Resize(spatial_size=[64,64,64], mode='nearest'),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]),
    'val': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Resize(spatial_size=[64,64,64], mode='nearest'),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]),
    'test': Compose([
        LoadImage(),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
}

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
    
tumor_seg_transform = {
    'train':  Compose([
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
        # RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandSpatialCropd(keys=["image", "label"], roi_size=[64, 64, 64], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]),
    'val': Compose([
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
    ])
}