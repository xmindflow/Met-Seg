import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep, TransformBackends
from monai import data
import monai.transforms as T
from dataset.bratsMets_detection import (
    load_brats_mets_dataset_paths,
    get_brats_mets_transforms,
)
import pandas as pd

join = os.path.join


def get_brats_mets_detection_dataloader(
    config: EasyDict, modes: Tuple[str, str, str] = ("tr", "vl", "te")
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    images_paths = load_brats_mets_dataset_paths(config.dataset.data_root, config.dataset.get("modalities",None))
    train_transform, val_transform, test_transform = get_brats_mets_transforms(config)

    ######################### loading data depending on the split files
    patient_names = {}
    path = (
        os.getcwd()
        if os.path.basename(os.getcwd()) != "src"
        else os.path.dirname(os.getcwd())
    )
    for key in ["train", "validation", "test"]:
        df = pd.read_csv(os.path.abspath(f"{path}/data_analysis/{key}_patients.csv"))
        names = df[df.columns[0]].tolist()
        patient_names[key] = names

    splitted_image_paths = {"train": [], "validation": [], "test": []}
    for path in images_paths:
        patient_name = path["name"]
        if patient_name in patient_names["train"]:
            splitted_image_paths["train"].append(path)
        elif patient_name in patient_names["validation"]:
            splitted_image_paths["validation"].append(path)
        elif patient_name in patient_names["test"]:
            splitted_image_paths["test"].append(path)
        else:
            raise ValueError(f"Patient {patient_name} not found in any of the splits")

    ######################### loading data depending on the split files
    loaders = []
    use_cache = config.data_loader.get("use_cache", False)
    dataset_class = data.CacheDataset if use_cache else data.Dataset

    for mode, transform, paths, loader_config in zip(
        ["tr", "vl", "te"],
        [train_transform, val_transform, test_transform],
        [splitted_image_paths["train"], splitted_image_paths["validation"], splitted_image_paths["test"]],
        [config.data_loader.train, config.data_loader.validation, config.data_loader.test]
    ):
        if mode in modes:
            if use_cache:
                dataset = dataset_class(data=paths, transform=transform, num_workers=16)
            else:
                dataset = dataset_class(data=paths, transform=transform)
            loader = data.DataLoader(dataset, **loader_config)
            loaders.append(loader)
            
    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
