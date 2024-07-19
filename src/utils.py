class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# from termcolor import colored
import torch
import torch.nn.functional as F
import torchvision
import os
from monai.transforms import (
    ResizeWithPadOrCropd,
)
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor
import platform
from tqdm import tqdm
import h5py
import pandas as pd
from joblib import Parallel, delayed
import argparse
from brats_metrics.metrics import get_LesionWiseResults
import json
import yaml


def get_parser():
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        "-c",
        "--config",
        help="configuration file *.yml",
        type=str,
        required=True,
        default="",
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="seed for reproducibility",
        type=int,
        required=False,
        default=-1,
    )

    # training parameters
    parser.add_argument(
        "-eps",
        "--epochs",
        help="num of epochs for train",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument(
        "-iters",
        "--iterations",
        help="num of iterations for train",
        type=int,
        required=False,
        default=-1,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="learning rate",
        type=float,
        required=False,
        default=-1,
    )
    parser.add_argument(
        "-b", "--batch", help="batch size", type=int, required=False, default=-1
    )
    return parser


def update_config_with_args(config, args):
    if args.learning_rate != -1:
        config.optimizer.params.lr = args.learning_rate
    if args.iterations != -1:
        config.trainer.max_steps = args.iterations
    if args.epochs != -1:
        config.trainer.max_epochs = args.epochs
    if args.batch != -1:
        config.data_loader.train.batch_size = args.batch
    return config


def InverseConvertToMultiChannelBasedOnBratsClasses(label: torch.Tensor):
    assert label.ndim == 5  # (batch, classes, x, y, z)
    assert label.shape[1] == 3
    # WT TC ET
    result = torch.zeros_like(label[:, 0:1], dtype=torch.uint8)
    result[label[:, 2:3] == 1] = 3
    result[(label[:, 1:2] == 1) & (label[:, 2:3] != 1)] = 1
    result[(label[:, 0:1] == 1) & (label[:, 1:2] != 1) & (label[:, 2:3] != 1)] = 2
    return result


def _print(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{bcolors.ENDC}"

    if "bold" in p.lower():
        pre += bcolors.BOLD
    elif "underline" in p.lower():
        pre += bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += bcolors.HEADER

    if "warning" in p.lower():
        pre += bcolors.WARNING
    elif "error" in p.lower():
        pre += bcolors.FAIL
    elif "ok" in p.lower():
        pre += bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += bcolors.OKBLUE
        else:
            pre += bcolors.OKCYAN

    print(f"{pre}{string}{bcolors.ENDC}")


def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            expanded_config = expand_env_vars_in_data(config)
            return expanded_config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)


def expand_env_vars_in_data(data):
    """
    Recursively walk through the data structure,
    expanding environment variables in string values.
    """
    if isinstance(data, dict):
        # For dictionaries, apply expansion to each value
        return {key: expand_env_vars_in_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        # For lists, apply expansion to each element
        return [expand_env_vars_in_data(element) for element in data]
    elif isinstance(data, str):
        # For strings, expand environment variables
        return os.path.expandvars(data)
    else:
        # For all other data types, return as is
        return data


def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        _print("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")


def cal_lesion_based_metric(nifty_dir=None):
    if nifty_dir is None:
        raise ValueError("nifty_dir is None")
    preds_list = sorted(
        glob.glob(os.path.join(nifty_dir, "BraTS-MET*", "*_pred.nii.gz"))
    )
    seg_list = sorted(glob.glob(os.path.join(nifty_dir, "BraTS-MET*", "*_gt.nii.gz")))
    if not preds_list or not seg_list:
        raise ValueError("preds_list or seg_list is empty")
    if len(preds_list) != len(seg_list):
        raise ValueError("preds_list and seg_list have different length")
    results = process_in_parallel(preds_list, seg_list)
    print("Saving the results!!!!!!")
    save_lesion_based_results(results, nifty_dir)
    mean_metrics = calculate_metrics_mean(results)
    # for key in mean_metrics["dice"].keys():
    #     print(f"Lesion wise dice score for <{key}> is {mean_metrics['dice'][key]:.3f}")
    #     print(f"Lesion wise hd95 score for <{key}> is {mean_metrics['hd'][key]:.3f}")
    #     print(
    #         f"Legacy dice score for <{key}> is {mean_metrics['legacy_dice'][key]:.3f}"
    #     )
    #     print(f"Legacy hd95 score for <{key}> is {mean_metrics['legacy_hd'][key]:.3f}")
    return mean_metrics


def save_lesion_based_results(results: list, nifty_dir: str):
    patient_list = sorted(glob.glob(os.path.join(nifty_dir, "BraTS-MET*")))
    patient_names = [os.path.basename(patient) for patient in patient_list]
    with pd.ExcelWriter(os.path.join(nifty_dir, "results_per_patient.xlsx")) as writer:
        for i, df in enumerate(results):
            df.to_excel(writer, sheet_name=f"{patient_names[i]}")


def calculate_metrics_mean(results):
    metrics = {
        "dice": {"wt": [], "tc": [], "et": []},
        "hd": {"wt": [], "tc": [], "et": []},
        "legacy_dice": {"wt": [], "tc": [], "et": []},
        "legacy_hd": {"wt": [], "tc": [], "et": []},
    }

    for df in results:
        for key in metrics["dice"].keys():
            for metric_type, metric_name in zip(
                ["dice", "hd", "legacy_dice", "legacy_hd"],
                [
                    "LesionWise_Score_Dice",
                    "LesionWise_Score_HD95",
                    "Legacy_Dice",
                    "Legacy_HD95",
                ],
            ):
                metrics[metric_type][key].append(
                    df[df["Labels"] == key.upper()][metric_name].values[0]
                )

    mean_metrics = {
        metric_type: {key: np.mean(values) for key, values in metric_dict.items()}
        for metric_type, metric_dict in metrics.items()
    }

    return mean_metrics


def process_in_parallel(preds_list, seg_list):
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(get_LesionWiseResults, preds_list, seg_list),
    #             total=len(preds_list),
    #             desc="calculating lesion wise results",
    #         )
    #     )
    results = Parallel(n_jobs=-1)(
        delayed(get_LesionWiseResults)(pred, seg)
        for pred, seg in tqdm(
            zip(preds_list, seg_list), desc="calculating lesion wise results"
        )
    )
    return results
