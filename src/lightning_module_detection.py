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
from metrics import get_binary_metrics_detection
import torchmetrics
from PIL import Image
import matplotlib.pyplot as plt


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

        self.metrics = {}
        for mode in modes:
            self.metrics[mode] = get_binary_metrics_detection(mode=mode).clone(
                prefix=f"metrics_{self.modes_dict[mode]}/"
            )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="binary", num_classes=2, validate_args=False, normalize="true",# threshold=0.5508
        )
        self.PR_curve = torchmetrics.PrecisionRecallCurve(
            task="binary",
            validate_args=False,  # thresholds= 10
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

        self.post_trans = T.Compose(
            [
                T.Activations(sigmoid=True),
                T.AsDiscrete(threshold=0.5),
            ]
        )

        self.model_name = config.model.name.split("_")[0]
        self.save_hyperparameters(config)

        #### a bug in the pytorch lightning that the on_epoch_end is called in the begining of the training when resuming from checkpoints####
        self.pass_in_the_begining = False
        if config.checkpoints.continue_training:
            self.pass_in_the_begining = True

        ########### for the situation for validation and test we need to have teh shapes of the before and after the padding of teh splitter ######
        self.original_shape = None
        self.padded_shape = None

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

        if stage is not "tr":
            self.msk = msks.view(
                msks.shape[2:]
            )  # we need to save it for just evaluation
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
            imgs = torch.stack(temp_imgs).squeeze(
                1
            )  # because we would have (batch, original_batch=1,channels, height, width, depth) and we need to remove the original batch
            msks = torch.stack(temp_msks).squeeze(1)

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

        self._compute_and_log_type_metrics(stage)

        if stage not in ["tr"]:
            self._save_confusion_matrix_plot(stage)
            self._save_PR_curve_plot(stage)           
                
        if stage == "tr":
            if not self.config.trainer.enable_progress_bar:
                print(f"Epoch {self.current_epoch} is done")


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

    def _save_PR_curve_plot(self, stage: str):
        precision, recall, thresholds = self.PR_curve.compute()
        # print(f"precision: {precision}")
        # print(f"recall: {recall}")
        # print(f"thresholds: {thresholds}")
        fig_, ax_ = self.PR_curve.plot(score=True)
        fig_.canvas.draw()
        data = np.fromstring(fig_.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig_.canvas.get_width_height()[::-1] + (3,))
        self.logger.experiment.add_image(
            f"PR Curve/{self.modes_dict[stage]}",
            data,
            global_step=self.current_epoch,
            dataformats="HWC",
        )
        # plt.close(fig_)
        self.PR_curve.reset()
        f1 = 2 * (precision * recall) / (precision + recall)
        # print(f"f1: {f1}")
        f1_max, idx = f1.max(), f1.argmax()
        # print(f"f1_max: {f1_max}")
        # print(f"thresholds[idx]: {thresholds[idx]}")
        # print(f"idx: {idx}")

        self.log(f"max_f1_{self.modes_dict[stage]}", f1_max)
        self.log(f"max_threshold_{self.modes_dict[stage]}", thresholds[idx])
        # exit()

    def _compute_and_log_type_metrics(self, stage: str):
        """Compute, log, and reset metrics."""
        metric = self.metrics[stage].compute()
        self.log_dict({f"{k}": v.mean() for k, v in metric.items()})
        if stage in ["vl"]:
            self.log(
                f"total_f1_{self.modes_dict[stage]}",
                metric[f"metrics_{self.modes_dict[stage]}/BinaryF1Score"].mean(),
            )
        self.metrics[stage].reset()


    def _shared_step(self, batch, stage: str, batch_idx=None) -> torch.Tensor:
        imgs, gts, imgs_loc, gts_loc = self._extract_data(batch, stage)
        # print(f"imgs shape: {imgs.shape}")
        # print(f"gts shape: {gts.shape}")
        # print(f"gts.metadata: {gts.meta}")
        preds = self._forward_pass(imgs, stage)
        # print(f"preds shape: {preds.shape}")
        # print(f"preds.metadata: {preds.meta}")
        gts_transformed, preds_transformed = self._transform_label_and_prediction(
            gts, preds
        )
        # print(f"gts_binary shape: {gts.shape}")
        loss = self._cal_losses(preds_transformed, gts_transformed, stage)
        self._update_metrics(preds_transformed, gts_transformed, stage)
        return loss

    # batch have values of size (batch ,channels, height, width, depth)
    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "tr", batch_idx)}

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
        total_loss = torch.tensor(0.0, device=preds.device)

        # if stage is not "tr": # when calculating the loss for the validation and test we need to have the batch size = 1
        #     gts = gts.unsqueeze(0)
        #     preds = preds.unsqueeze(0)

        for loss_name, (loss_cls, coef) in self.critrion.items():
            loss = loss_cls(preds, gts)
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

    def _transform_label_and_prediction(self, label_batch, pred_batch):
        """
        Transforms a batch of labels from shape (b, c, h, w, d) to (b, c, 1, 1, 1).
        Each output element is 1 if the sum of the corresponding label > 1, otherwise 0.

        Args:
        label_batch (torch.Tensor): A batch of labels with shape (b, c, h, w, d).

        Returns:
        torch.Tensor: Transformed labels with shape (b, c, 1).
        """
        # Sum the labels across the spatial dimensions (h, w, d)
        label_sums = label_batch.sum(dim=(2, 3, 4), keepdim=True)

        # Convert sums to binary: 1 if sum > 1, else 0
        binary_labels = (
            label_sums > 1
        ).float()  # Use `.int()` to convert from bool to int

        pred_batch_unsqueeze = pred_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return binary_labels, pred_batch_unsqueeze

    def _update_metrics(
        self, preds: torch.Tensor, gts: torch.Tensor, stage: str
    ) -> None:
        metrics = self.metrics[stage]
        pred = preds  # since the output is logits we convert it to sigmoid and then to binary
        gt = gts.type(torch.uint8)
        metrics.to(self.device)
        if stage in ["vl", "te"]:
            # when calculating the metrics for the validation and test we need to have the batch size = 1
            gt = gt.unsqueeze(0)
            pred = pred.unsqueeze(0)
            self.confusion_matrix.update(pred, gt)
            self.PR_curve.update(pred, gt)
        metrics.update(pred, gt)

    def _get_center_crop_coords(self, patch_slices):
        center_crop_slices = []
        for sl in patch_slices:
            # Calculate the midpoint of the slice
            midpoint = (sl.stop + sl.start) // 2
            # Calculate half the size of the slice's length
            half_length = (
                sl.stop - sl.start
            ) // 4  # because we want half of half (center crop)
            # Create new slice for the center crop
            crop_start = midpoint - half_length
            crop_stop = midpoint + half_length
            center_crop_slices.append(slice(crop_start, crop_stop))
        return center_crop_slices

    def _calculate_slice_location(self, loc, offsets):
        slice_loc = []
        for lo, of in zip(loc, offsets):
            slice_loc.append(slice(lo, lo + of))
        return tuple(slice_loc)

    def save_nifty(self, gt_predicted, gt, patient_name, stage):
        save_dir = os.path.join(self.logger.log_dir, f"nifty_{stage}")
        os.makedirs(os.path.join(save_dir, patient_name), exist_ok=True)
        pred_nifty = nib.Nifti1Image(gt_predicted.detach().cpu().numpy(), np.eye(4))
        gt_nifty = nib.Nifti1Image(gt.detach().cpu().numpy(), np.eye(4))
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


    def return_model(self):
        return self.model
