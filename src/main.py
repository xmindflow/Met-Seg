import os
import time
import glob
import warnings
import pytorch_lightning as pl
import numpy as np
from matplotlib import pyplot as plt
from torchinfo import summary
import yaml
import platform
import torch
from monai.utils.misc import set_determinism
from monai.utils import ensure_tuple_rep

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner


from loader.bratsMets_detection import get_brats_mets_detection_dataloader
from loader.bratsMets_detect_seg import get_brats_mets_detect_seg_dataloader
from utils import load_config, print_config, get_parser, update_config_with_args
from models.get_models import get_model
from lightning_module_detection import module as lightning_module_detection
from lightning_module_detect_seg import module as lightning_module_detect_seg
from easydict import EasyDict

warnings.filterwarnings("ignore")


def set_seed(seed):
    if seed != -1:
        set_determinism(seed=seed)
        pl.seed_everything(seed=seed)


def configure_logger(config, path, run_name):
    path = os.path.join(path, "tb_logs", config.model.name)
    return TensorBoardLogger(path, name=run_name)


def load_config_file(config_file: str):
    path = (
        os.getcwd()
        if os.path.basename(os.getcwd()) != "src"
        else os.path.dirname(os.getcwd())
    )

    CONFIG_FILE_PATH = os.path.abspath(
        os.path.join(path, "config", config_file, f"{config_file}.yml")
    )
    config = load_config(CONFIG_FILE_PATH)
    print_config(config)
    config = EasyDict(config)
    return config


def configure_trainer(config, logger):
    if config.get("detect", False) or config.get("detect_mask", False):
        checkpoint_callback = ModelCheckpoint(
            monitor="total_f1_val",
            dirpath=logger.log_dir,
            filename=f"{config.model.name}-{{epoch:02d}}-{{total_f1_val:.6f}}",
            save_top_k=3,
            mode="max",
            save_last=True,
        )
    elif config.get("detect_seg", False) or config.get("detect_seg_mask", False):
        checkpoint_callback = ModelCheckpoint(
            monitor="total_dice_val",
            dirpath=logger.log_dir,
            filename=f"{config.model.name}-{{epoch:02d}}-{{total_dice_val:.6f}}",
            save_top_k=3,
            mode="max",
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="total_dice_val",
            dirpath=logger.log_dir,
            filename=f"{config.model.name}-{{epoch:02d}}-{{total_dice_val:.6f}}",
            save_top_k=3,
            mode="max",
            save_last=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        **config.trainer,
    )


def get_lightning_and_loader_class(config):
    print(config.model.name.lower())
    if config.get("detect", False):
        print("Using the detection model")
        return lightning_module_detection, get_brats_mets_detection_dataloader
    elif config.get("detect_seg", False):
        print("Using the detection and segmentation model")
        return lightning_module_detect_seg, get_brats_mets_detect_seg_dataloader
    else:
        raise NotImplementedError("The mode is not implemented yet!!!!!!!!!!!!!!!!!!")


def main():
    ######################## parse the arguments ########################
    parser = get_parser()
    args = parser.parse_args()

    ######################## load the configuration file ########################
    set_seed(args.seed)

    config = load_config_file(args.config)

    lightning_module_cls, loader_func = get_lightning_and_loader_class(config)

    tr_loader, vl_loader = loader_func(
        config, ("tr", "vl")
    )  

    if config.trainer.get("max_epochs", -1) == -1:
        config.trainer.max_epochs = config.trainer.max_steps // len(tr_loader) + 1

    config = update_config_with_args(config, args)

    print(f"epochs: {config.trainer.max_epochs}")
    ######################### get the model and configure the trainer ########################

    network = get_model(config)
    if config.get("detect_seg_mask", False) or config.get("detect_mask", False):
        shape = config.dataset.mask_size
    else:
        shape = config.dataset.input_size

    len_modalities = len(config.dataset.get("modalities", [0, 0, 0, 0]))       
    summary(
        network,
        input_size=(config.data_loader.train.batch_size, len_modalities, *shape),
        col_names=["input_size", "output_size", "num_params", "mult_adds", "trainable"],
        depth=3,
        mode="eval",
    )
    # generate a name for the model based on the time and name of the model
    custom_name = (
        f"_{config.model.customized_name}"
        if config.model.get("customized_name", None) is not None
        else ""
    )
    RUN_NAME = (
        f"{config.model.name}{custom_name}"  # _{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    logger = configure_logger(config, config.results_path, RUN_NAME)
    trainer = configure_trainer(config, logger)

    ######################### train the model ########################
    test_mode = config.checkpoints.test_mode
    lr_find = config.get("lr_find", False)

    ############################### load the checkpoint if exists ###############################
    ckpt_path = None
    if config.checkpoints.continue_training or test_mode:
        ckpt_path = config.checkpoints.ckpt_path
        if not os.path.exists(ckpt_path):
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')

    if not test_mode:
        model = lightning_module_cls(config, model=network)
        if lr_find:
            Tuner(trainer).lr_find(
                model, train_dataloaders=tr_loader, val_dataloaders=vl_loader
            )
            config.optimizer.params.lr = model.lr

            # lr_find_files = glob.glob(".lr_find*")
            # if lr_find_files:
            #     os.remove(lr_find_files[0])

        ############################### save the configuration file ###############################
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(
            os.path.join(logger.log_dir, "hpram_by_yousef.yaml"), "w"
        ) as yaml_file:
            yaml.dump(config, yaml_file)
        ######################### train the model ########################
        trainer.fit(
            model,
            train_dataloaders=tr_loader,
            val_dataloaders=vl_loader,
            ckpt_path=ckpt_path,
        )

        ######################### test the model ########################
        print(f"testing the model from the best checkpoint of model name: {RUN_NAME}")
        te_loader = loader_func(config, ("te"))
        trainer.test(
            model, dataloaders=te_loader, ckpt_path="best"
        )  # uses the best model to do the test

    ######################### test the model ########################
    else:
        print(f"Try to test from {ckpt_path}")
        try:
            model = lightning_module_cls.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                map_location="cpu",
                config=config,
                model=network,
            )
        except:
            checkpoint = torch.load(ckpt_path)
            network.load_state_dict(checkpoint["MODEL_STATE"], strict=False)
            model = lightning_module_cls(config, model=network)

        te_loader = loader_func(config, ("te"))
        print("loaded the checkpoint.")
        trainer.test(model, dataloaders=te_loader)


if __name__ == "__main__":
    main()
