import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import nibabel as nib
from monai.losses import DiceLoss, TverskyLoss, FocalLoss, MaskedDiceLoss, MaskedLoss
from torch.nn import BCEWithLogitsLoss
from monai import transforms as T
from monai.inferers import SlidingWindowInferer, SlidingWindowSplitter
from monai.utils import ensure_tuple_rep
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optimizers import *
from typing import Union, Tuple, Dict
from easydict import EasyDict
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from metrics import get_binary_metrics, get_binary_metrics_detection
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
from typing import Optional, Any


class MaskedBCEWithLogitsLoss(torch.nn.Module):
    """This loss combines a Sigmoid layers and a masked BCE loss in one single
    class. It's AMP-eligible.

    Args:
        eps (float): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.loss = BCEWithLogitsLoss(reduction="none")

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """

        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.ndim == gt.ndim
        assert mask.shape[-3:] == gt.shape[-3:]  # B, C, H, W, D

        assert gt.max() <= 1 and gt.min() >= 0
        loss = self.loss(pred, gt)

        return (loss * mask).sum() / (mask.sum() + self.eps)


class MaskedFocalLoss(torch.nn.Module):
    """This loss combines a Sigmoid layers and a masked Focal loss in one single
    class. It's AMP-eligible.

    Args:
        gamma (float): Gamma parameter for the Focal loss. Defaults to 2.0.
        eps (float): Eps to avoid zero-division error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.eps = eps
        self.loss = FocalLoss(reduction="none", **kwargs)

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """

        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.ndim == gt.ndim
        assert mask.shape[-3:] == gt.shape[-3:]  # B, C, H, W, D

        assert gt.max() <= 1 and gt.min() >= 0
        loss = self.loss(pred, gt)

        return (loss * mask).sum() / (mask.sum() + self.eps)


class MaskedTverskyLoss(TverskyLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.tverskyloss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = MaskedLoss(loss=super().forward)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)  # type: ignore[no-any-return]
