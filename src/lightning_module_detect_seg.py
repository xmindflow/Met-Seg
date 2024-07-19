import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import nibabel as nib
from monai.losses import DiceLoss, TverskyLoss, FocalLoss
from torch.nn import BCEWithLogitsLoss
from monai import transforms as T
from monai.inferers import SlidingWindowSplitter
from monai.utils import ensure_tuple_rep
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optimizers import *
from typing import Union, Tuple, Dict
from easydict import EasyDict
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from metrics import get_binary_metrics
from utils import (
    InverseConvertToMultiChannelBasedOnBratsClasses,
    cal_lesion_based_metric,
)
import torchmetrics
from PIL import Image
import matplotlib.pyplot as plt
from models.get_models import get_model
from lightning_module_detection import module as detection_module
import torch.nn.functional as F


class module(pl.LightningModule):
    def __init__(self, config: EasyDict, model=None):
        super(module, self).__init__()
        ############### initilizing the model and the config ################
        self.config = config
        self.model = model
        ############### initilizing the loss function ################
        self.get_losses(self.config)
        if len(self.critrion) == 0:
            raise ValueError("No loss function is defined")
        self.print_losses()
        ############### initilizing the metrics ################
        modes = ["tr", "vl", "te"]
        self.modes_dict = {"tr": "train", "vl": "val", "te": "test"}
        self.types = ["wt", "tc", "et"]
        self.metrics = {}
        for mode in modes:
            self.metrics[mode] = {}
            for type_ in self.types:
                self.metrics[mode][type_] = get_binary_metrics(mode=mode).clone(
                    prefix=f"metrics_{self.modes_dict[mode]}_{type_}/"
                )
        ################# calculation of number of params and FLOPS ################
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        # self.log("n_parameters_M", round(n_parameters * 1e-6, 2))
        size = config.dataset.input_size
        input_res = (len(config.dataset.get("modalities", [0, 0, 0, 0])) , *size)
        input = torch.ones(()).new_empty(
            (1, *input_res),
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )
        flops = FlopCountAnalysis(self.model, input)
        total_flops = flops.total()
        print(f"MAdds: {round(total_flops * 1e-9, 2)} G")
        # self.log("GFLOPS", round(total_flops * 1e-9, 2))

        # ##### calculate the params and FLOPs (ptflops)
        flops, params = get_model_complexity_info(
            self.model, input_res, as_strings=True, print_per_layer_stat=False
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", flops))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
        # self.log("n_parameters_M", params)
        # self.log("GFLOPS", flops)

        self.lr = config.optimizer.params.lr

        self.splitter = SlidingWindowSplitter(
            patch_size=size,
            **config.sliding_window_params,
        )

        # self.post_trans = T.Compose(
        #     [
        #         T.Activations(sigmoid=True),
        #         T.AsDiscrete(threshold=0.5),
        #     ]
        # )

        self.model_name = config.model.name.split("_")[0]
        self.save_hyperparameters(config)

        #### a bug in the pytorch lightning that the on_epoch_end is called in the begining of the training when resuming from checkpoints####
        self.pass_in_the_begining = False
        if config.checkpoints.continue_training:
            self.pass_in_the_begining = True

        ########### for the situation for validation and test we need to have teh shapes of the before and after the padding of teh splitter ######
        self.original_shape = None
        self.padded_shape = None

        ########## load the detection model ##########
        detector_module = detection_module.load_from_checkpoint(
            config.model_detection.ckpt_path,
            map_location="cpu",
            config=config,
            model=get_model(config.model_detection),
        )
        self.detector = detector_module.return_model().cuda()
        del detector_module
        print("*" * 50)
        print("The detector model is loaded successfully")
        print("*" * 50)

    def forward(self, x):
        return self.model(x)

    def get_losses(self, config):
        losses_name = {
            "bce": BCEWithLogitsLoss,
            "dice": DiceLoss,
            "tversky": TverskyLoss,
            "focal": FocalLoss,
        }
        self.critrion = {}
        for loss_name, values in config.criterion.items():
            if loss_name in losses_name and values["coef"] > 0:
                self.critrion[f"{loss_name}_loss"] = (
                    losses_name[loss_name](**values.get("params", {})),
                    values["coef"],
                )

    def print_losses(self):
        print("#" * 50, "  LOSSES  ", "#" * 50)
        for loss_name, (loss_cls, coef) in self.critrion.items():
            print(f"{loss_name} with coef: {coef}")
        print("#" * 50)

    def _extract_data(
        self, batch: dict, stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"image shape: {batch['image'].shape}")
        # print(f"label shape: {batch['label'].shape}")
        imgs = batch["image"]  # .float()
        msks = batch["label"]  # .type(torch.uint8)
        imgs_loc = []
        msks_loc = []
        # print(f"image shape: {imgs.shape}")
        # print(f"label shape: {msks.shape}")
        if stage is not "tr":
            self.msk = msks  # we need to save it for just evaluation
            imgs_splitted = self.splitter(imgs)
            msks_splitted = self.splitter(msks)
            # we need the pad sizes to get back to the original shape
            self._get_pad_size(imgs)
            self.original_shape = self.splitter.get_input_shape(imgs)
            self.padded_shape = self.splitter.get_padded_shape(imgs)
            temp_imgs = []
            temp_msks = []
            for im, lab in zip(imgs_splitted, msks_splitted):
                imgs_loc.append(im[1])
                msks_loc.append(lab[1])
                temp_imgs.append(im[0])
                temp_msks.append(lab[0])
            imgs = torch.cat(
                temp_imgs
            )  # because we would have (batch, original_batch=1,channels, height, width, depth) and we need to remove the original batch
            msks = torch.cat(temp_msks)

        return imgs, msks, imgs_loc, msks_loc

    def _get_pad_size(self, imgs):
        patch_size, overlap, offset = self.splitter._get_valid_shape_parameters(
            imgs.shape[2:]
        )
        pad_size, _ = self.splitter._calculate_pad_size(
            imgs.shape[2:], 3, patch_size, offset, overlap
        )
        self.pad_start = pad_size[1::2]
        self.pad_end = pad_size[::2]

    def on_epoch_end(self, stage: str):
        if (
            self.pass_in_the_begining and stage == "tr"
        ):  # to avoid the bug in the pytorch lightning
            self.pass_in_the_begining = False
            return

        if stage not in ["tr"]:
            total_dice = [
                self._compute_and_log_type_metrics(stage, type_) for type_ in self.types
            ]
            # Log the mean of total Dice scores.
            self.log(f"total_dice_{self.modes_dict[stage]}", torch.stack(total_dice).mean())

        if stage in ["te"]:
            self._log_brats_metrics_for_stage(stage)

        if stage == "tr":
            if not self.config.trainer.enable_progress_bar:
                print(f"Epoch {self.current_epoch} is done")

    def _log_brats_metrics_for_stage(self, stage: str) -> None:
        brats_metric = cal_lesion_based_metric(
            os.path.join(self.logger.log_dir, f"nifty_{stage}")
        )
        self._log_brats_metric(brats_metric, stage)

    def _save_confusion_matrix_plot(self, stage: str):
        print(self.confusion_matrix.compute())
        fig_, ax_ = self.confusion_matrix.plot()
        fig_.canvas.draw()
        data = np.fromstring(fig_.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig_.canvas.get_width_height()[::-1] + (3,))
        self.logger.experiment.add_image(
            f"Confusion Matrix/{self.modes_dict[stage]}",
            data,
            global_step=self.current_epoch,
            dataformats="HWC",
        )
        # plt.close(fig_)
        self.confusion_matrix.reset()

    def _compute_and_log_type_metrics(self, stage: str, type_: str) -> torch.Tensor:
        """Compute, log, and reset metrics for a given type, returning the Dice score."""
        metric = self.metrics[stage][type_].compute()
        self.log_dict({f"{k}": v.mean() for k, v in metric.items()})
        dice_score = metric[f"metrics_{self.modes_dict[stage]}_{type_}/Dice"]
        self.metrics[stage][type_].reset()
        return dice_score


    def _shared_step(self, batch, stage: str, batch_idx=None) -> torch.Tensor:
        imgs, gts, imgs_loc, gts_loc = self._extract_data(batch, stage)
        ########### we need to do the detection first ###########
        self.detector.eval()
        with torch.inference_mode():
            preds_detected = self.detector(imgs)
        ########### get the images that have the tumor ###########
        chosen_imgs, chosen_gts, chosen_imgs_loc, chosen_gts_loc = [], [], [], []
        for i in range(len(imgs)):
            if preds_detected[i] > 0:
                chosen_imgs.append(imgs[i])
                chosen_gts.append(gts[i])
                if stage not in ["tr"]:
                    chosen_imgs_loc.append(imgs_loc[i])
                    chosen_gts_loc.append(gts_loc[i])
        ########### if there is no tumor in the image we return 0 loss ###########
        if len(chosen_imgs) == 0:
            return torch.tensor(0.0, device=imgs.device, requires_grad=True)

        # print(f"len(chosen_imgs): {len(chosen_imgs)}")
        chosen_imgs = torch.stack(chosen_imgs)
        chosen_gts = torch.stack(chosen_gts)
        ########### now we can do the segmentation ###########
        preds_chosen = self._forward_pass(chosen_imgs, stage)
        # loss= self._cal_losses(preds_chosen, chosen_gts, stage)
        # self._update_metrics(preds_chosen, chosen_gts, stage)
        if stage not in ["tr"]:
            pred_img_logits = self._generate_prediction_image(
                preds_chosen, chosen_gts_loc
            )
            loss = self._cal_losses(pred_img_logits, self.msk.float(), stage)
            self._update_metrics(pred_img_logits, self.msk, stage)
            self.save_nifty(pred_img_logits, self.msk, batch["name"][0], stage)
        return loss

    def _generate_prediction_image(self, preds, gts_loc):
        pred_img = torch.zeros((3, *self.padded_shape), device=preds.device)
        pred_counts = torch.zeros((3, *self.padded_shape), device=preds.device)
        for pred, loc in zip(preds, gts_loc):
            slice_loc = self._calculate_slice_location(loc, pred.shape[1:])
            pred_img[:, *slice_loc] += pred
            pred_counts[:, *slice_loc] += 1

        end_slices = [
            -self.pad_end[i] if self.pad_end[i] != 0 else None for i in range(3)
        ]
        pred_img_crop = pred_img[
            :,
            self.pad_start[0] : end_slices[0],
            self.pad_start[1] : end_slices[1],
            self.pad_start[2] : end_slices[2],
        ]
        pred_counts_crop = pred_counts[
            :,
            self.pad_start[0] : end_slices[0],
            self.pad_start[1] : end_slices[1],
            self.pad_start[2] : end_slices[2],
        ]
        # make sure that the places that the count is zero to become 1 to avoid division by zero
        pred_counts_crop[pred_counts_crop == 0] = 1
        # make sure that the places that are zero to be less than zero so that in thresholding we can have zero
        pred_img_crop[pred_img_crop == 0] = -1
        pred_img_crop.div_(pred_counts_crop)
        pred_img_crop = pred_img_crop.unsqueeze(0)  # give it a batch size of 1
        # print(f"pred_img_crop shape: {pred_img_crop.shape}")
        # print(f"self.msk shape: {self.msk.shape}")
        assert (
            pred_img_crop.shape == self.msk.shape
        )  # make sure that the shape is the same as the mask
        return pred_img_crop

    def _train_step(self, batch, stage: str, batch_idx=None) -> torch.Tensor:
        imgs, gts, imgs_loc, gts_loc = self._extract_data(batch, stage)
        ########### we need to do the detection first ###########
        self.detector.eval()
        with torch.inference_mode():
            preds_detected = self.detector(imgs)
        ########### get the images that have the tumor ###########
        chosen_imgs, chosen_gts, chosen_imgs_loc, chosen_gts_loc = [], [], [], []
        for i in range(len(imgs)):
            if preds_detected[i] > 0:
                chosen_imgs.append(imgs[i])
                chosen_gts.append(gts[i])
                if stage not in ["tr"]:
                    chosen_imgs_loc.append(imgs_loc[i])
                    chosen_gts_loc.append(gts_loc[i])
        ########### if there is no tumor in the image we return 0 loss ###########
        if len(chosen_imgs) == 0:
            return torch.tensor(0.0, device=imgs.device, requires_grad=True)

        # print(f"len(chosen_imgs): {len(chosen_imgs)}")

        chosen_imgs = torch.stack(chosen_imgs)
        chosen_gts = torch.stack(chosen_gts)
        ########### now we can do the segmentation ###########
        preds = self._forward_pass(chosen_imgs, stage)
        loss = self._cal_losses(preds, chosen_gts, stage)
        return loss

    # batch have values of size (batch ,channels, height, width, depth)
    def training_step(self, batch, batch_idx):
        return {"loss": self._train_step(batch, "tr", batch_idx)}

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("tr")

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._shared_step(batch, "vl", batch_idx)}

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("vl")

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._shared_step(batch, "te", batch_idx)}

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("te")

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.config.optimizer.name)
        ########### remove the lr from the optimizer params to avoid error in the optimizer initialization ####
        del self.config.optimizer.params["lr"]
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.lr,
            **self.config.optimizer.params,
        )
        scheduler_cls = globals().get(self.config.scheduler.name, None)
        if scheduler_cls is None:
            scheduler = None
        else:
            if hasattr(self.config.scheduler.params, "max_epochs"):
                self.config.scheduler.params.max_epochs = self.config.trainer.max_epochs
            scheduler = scheduler_cls(optimizer, **self.config.scheduler.params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_scheduler",
            },
        }

    def _cal_losses(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:

        # Determine if deep supervision is applied based on tensor dimensions
        is_deep_supervision = len(preds.size()) - len(gts.size()) == 1

        # Adjust predictions for deep supervision
        preds = torch.unbind(preds, dim=1) if is_deep_supervision else [preds]

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=gts.device)

        if gts.dtype not in [torch.float32, torch.float16, torch.float64]:
            gts = gts.type(torch.float32)

        # Calculate weighted loss for each prediction (considering deep supervision)
        for loss_name, (loss_cls, coef) in self.critrion.items():
            weighted_losses = (
                [0.5**i * loss_cls(pred, gts) for i, pred in enumerate(preds)]
                if is_deep_supervision
                else [loss_cls(preds[0], gts)]
            )
            loss = sum(weighted_losses)
            self.log(
                f"{self.modes_dict[stage]}_{loss_name}",
                loss,
                on_epoch=True,
                prog_bar=False,
            )
            total_loss += coef * loss

        # Log and return the summed total loss
        self.log(
            f"{self.modes_dict[stage]}_total_loss",
            total_loss,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def _update_metrics(
        self, preds: torch.Tensor, gts: torch.Tensor, stage: str
    ) -> None:
        is_deep_supervision = len(preds.size()) - len(gts.size()) == 1
        preds = torch.unbind(preds, dim=1) if is_deep_supervision else [preds]
        preds = (preds[0].sigmoid() > 0.5).type(
            torch.uint8
        )  # we only need the last layer
        gts = gts.type(torch.uint8)
        metrics = self.metrics[stage]
        for index, type_ in enumerate(self.types):
            pred = preds[:, index : index + 1]
            gt = gts[:, index : index + 1]
            metrics[type_].to(self.device)
            metrics[type_].update(pred, gt)

    def _calculate_slice_location(self, loc, offsets):
        slice_loc = []
        for lo, of in zip(loc, offsets):
            slice_loc.append(slice(lo, lo + of))
        return tuple(slice_loc)

    def save_nifty(self, preds, gts, patient_name, stage):
        save_dir = os.path.join(self.logger.log_dir, f"nifty_{stage}")
        final_preds = (F.sigmoid(preds) > 0.5).type(torch.uint8)
        gts = InverseConvertToMultiChannelBasedOnBratsClasses(gts)
        final_preds = InverseConvertToMultiChannelBasedOnBratsClasses(final_preds)

        os.makedirs(os.path.join(save_dir, patient_name), exist_ok=True)
        pred_nifty = nib.Nifti1Image(
            final_preds[0, 0].detach().cpu().numpy(), np.eye(4)
        )
        gt_nifty = nib.Nifti1Image(gts[0, 0].detach().cpu().numpy(), np.eye(4))
        nib.save(
            pred_nifty,
            os.path.join(save_dir, patient_name, f"{patient_name}_pred.nii.gz"),
        )
        nib.save(
            gt_nifty,
            os.path.join(save_dir, patient_name, f"{patient_name}_gt.nii.gz"),
        )

    def _forward_pass(self, imgs: torch.Tensor, stage: str) -> torch.Tensor | list:
        return self(imgs)  # if stage == "tr" else self.slider(imgs, self.model)

    def _log_brats_metric(self, brats_metric: dict, stage: str) -> None:
        total_lesion_based_dice = []
        for k, v in brats_metric["dice"].items():
            total_lesion_based_dice.append(v)
        self.log(
            f"total_lesion_based_dice_{self.modes_dict[stage]}",
            np.stack(total_lesion_based_dice).mean(),
        )

        self.log_dict(
            {
                f"metrics_lesion_based_{self.modes_dict[stage]}/{k.upper()}/Dice": v
                for k, v in brats_metric["dice"].items()
            }
        )
        self.log_dict(
            {
                f"metrics_lesion_based_{self.modes_dict[stage]}/{k.upper()}/HD95": v
                for k, v in brats_metric["hd"].items()
            }
        )
        totak_legacy_dice = []
        for k, v in brats_metric["legacy_dice"].items():
            totak_legacy_dice.append(v)
        self.log(
            f"total_legacy_dice_{self.modes_dict[stage]}",
            np.stack(totak_legacy_dice).mean(),
        )
        self.log_dict(
            {
                f"metrics_legacy_{self.modes_dict[stage]}/{k.upper()}/Dice": v
                for k, v in brats_metric["legacy_dice"].items()
            }
        )
        self.log_dict(
            {
                f"metrics_legacy_{self.modes_dict[stage]}/{k.upper()}/HD95": v
                for k, v in brats_metric["legacy_hd"].items()
            }
        )
