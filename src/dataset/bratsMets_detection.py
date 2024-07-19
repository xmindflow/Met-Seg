import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep, TransformBackends
from monai import config, data
import monai.transforms as T
from monai.transforms.compose import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize

join = os.path.join


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: config.KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: config.NdarrayOrTensor):
        # WT TC ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        # TC WT ET
        result = [
            (img == 1) | (img == 3) | (img == 2),
            (img == 1) | (img == 3),
            img == 3,
        ]
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, config.NdarrayOrTensor]
    ) -> Dict[Hashable, config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class BinarizeLabelDictTransform(MapTransform):
    """
    A transform to convert label arrays to a binary classification (0 or 1) based on the sum of their elements within a dictionary format.
    The transform assumes the input is a dictionary with a 'label' key.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: config.KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: config.NdarrayOrTensor):
        sum_img = np.sum(img) if isinstance(img, np.ndarray) else torch.sum(img)
        return 1 if sum_img > 1 else 0

    def __call__(
        self, data: Mapping[Hashable, config.NdarrayOrTensor]
    ) -> Dict[Hashable, config.NdarrayOrTensor]:
        # Iterate over each key in self.keys (in case there are multiple label keys)
        d = dict(data)
        for key in self.key_iterator(d):
            # Apply the binarization process to each specified key
            d[key] = self.converter(d[key])
        return d


# def load_brats_mets_dataset_paths(root):
#     images_path = os.listdir(root)
#     images_list = []
#     for path in images_path:
#         image_path = join(root, path)
#         flair_img = join(image_path, f"{path}-t2f.nii.gz")
#         t1_img = join(image_path, f"{path}-t1n.nii.gz")
#         t1ce_img = join(image_path, f"{path}-t1c.nii.gz")
#         t2_img = join(image_path, f"{path}-t2w.nii.gz")
#         seg_img = join(image_path, f"{path}-seg.nii.gz")
#         images_list.append(
#             {
#                 "image": [flair_img, t1_img, t1ce_img, t2_img],
#                 "label": seg_img,
#                 "name": path,
#             }
#         )
#     return images_list

def load_brats_mets_dataset_paths(root, modalities_list=None):
    if modalities_list is None:
        modalities_list = ["flair", "t1", "t1ce", "t2"]

    modality_to_extension = {
        "flair": "t2f",
        "t1": "t1n",
        "t1ce": "t1c",
        "t2": "t2w"
    }

    images_path = os.listdir(root)
    images_list = []

    for path in images_path:
        image_path = os.path.join(root, path)
        image_modalities_paths = []

        for modality in modalities_list:
            assert modality in modality_to_extension, f"Invalid modality: {modality}"
            image_modalities_paths.append(os.path.join(image_path, f"{path}-{modality_to_extension[modality]}.nii.gz"))

        seg_img = os.path.join(image_path, f"{path}-seg.nii.gz")

        images_list.append(
            {
                "image": image_modalities_paths,
                "label": seg_img,
                "name": path,
            }
        )

    return images_list


def get_brats_mets_transforms(
    config: EasyDict,
) -> Tuple[T.Compose, T.Compose, T.Compose]:
    train_transform = T.Compose(
        [
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.EnsureTyped(keys=["image", "label"]),
            T.Orientationd(keys=["image", "label"], axcodes="RAS"),
            T.CropForegroundd(
                keys=["image", "label"], source_key="image", allow_smaller=True
            ),
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # T.Spacingd(
            #     mode=("trilinear", "nearest"),
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     padding_mode="zeros",
            # ),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            T.RandAffined(
                keys=["image", "label"],
                rotate_range=(
                    (np.deg2rad(-30), np.deg2rad(30)),
                    (np.deg2rad(-30), np.deg2rad(30)),
                    (np.deg2rad(-30), np.deg2rad(30)),
                ),
                scale_range=((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)),
                padding_mode="zeros",
                mode=("trilinear", "nearest"),
                prob=0.2,
            ),
            T.RandGaussianNoised(keys="image", prob=0.1, mean=0, std=0.1),
            T.RandGaussianSmoothd(
                keys="image",
                prob=0.2,
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
            ),
            T.RandScaleIntensityd(
                keys="image", factors=0.25, prob=0.15, channel_wise=True
            ),
            T.RandScaleIntensityFixedMeand(
                keys="image",
                factors=0.25,
                prob=0.15,
                fixed_mean=True,
                preserve_range=True,
            ),
            T.RandSimulateLowResolutiond(
                keys="image",
                prob=0.25,
            ),
            T.RandAdjustContrastd(
                keys="image",
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
                prob=0.1,
            ),
            T.RandAdjustContrastd(
                keys="image",
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                prob=0.3,
            ),
            T.RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            T.RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            T.RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            T.SpatialPadd(
                keys=["image", "label"],
                spatial_size=ensure_tuple_rep(config.dataset.input_size, 3)
                if config.get("detect", True)
                else ensure_tuple_rep(config.dataset.mask_size, 3),
            ),
            T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=ensure_tuple_rep(config.dataset.input_size, 3)
                if config.detect
                else ensure_tuple_rep(config.dataset.mask_size, 3),
                pos=config.dataset.pos_sample_num,
                neg=config.dataset.neg_sample_num,
                num_samples=config.dataset.num_samples_per_patient,
                image_key="image",
                image_threshold=0,
            ),
            T.CastToTyped(keys=["image", "label"], dtype=(np.float32, np.float32)),
            T.ToNumpyd(keys=["image", "label"]),
        ]
    )
    val_transform = T.Compose(
        [
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.EnsureTyped(keys=["image", "label"]),
            T.Orientationd(keys=["image", "label"], axcodes="RAS"),
            T.CropForegroundd(
                keys=["image", "label"], source_key="image", allow_smaller=True
            ),
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # T.Spacingd(
            #     mode=("trilinear", "nearest"),
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     padding_mode="zeros",
            # ),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            T.CastToTyped(keys=["image", "label"], dtype=(np.float32, np.float32)),
        ]
    )

    test_transform = T.Compose(
        [
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.EnsureTyped(keys=["image", "label"]),
            T.Orientationd(keys=["image", "label"], axcodes="RAS"),
            T.CropForegroundd(
                keys=["image", "label"], source_key="image", allow_smaller=True
            ),
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # T.Spacingd(
            #     mode=("trilinear", "nearest"),
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     padding_mode="zeros",
            # ),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            T.CastToTyped(keys=["image", "label"], dtype=(np.float32, np.float32)),
        ]
    )

    return train_transform, val_transform, test_transform