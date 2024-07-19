import torchmetrics
import torch
import numpy as np
import torch.nn.functional as F
from medpy.metric.binary import hd95
from monai.metrics import HausdorffDistanceMetric


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_metrics(config, *args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Accuracy(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Dice(),
            torchmetrics.Precision(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Specificity(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            # IoU
            torchmetrics.JaccardIndex(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
        ],
        prefix="metrics/",
    )
    return metrics
    # test_metrics
    # test_metrics = metrics.clone(prefix="").to(device)


class MONAIHausdorffDistance(torchmetrics.Metric):
    def __init__(self, include_background=False, percentile=95):
        super().__init__()
        self.include_background = include_background
        self.percentile = percentile
        self.monai_hd95 = HausdorffDistanceMetric(
            include_background=include_background, percentile=percentile
        )

    def update(self, y_pred, y_true):
        # y_preds and y can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]). (we are using batch first)
        y_pred = y_pred.permute(0, 1, 3, 4, 2)
        y_true = y_true.permute(0, 1, 3, 4, 2)
        # Convert tensors to one-hot format
        y_pred = (F.sigmoid(y_pred) > 0.5).type(y_true.dtype)
        if y_pred.shape[1] == 1:
            y_pred_one_hot = torch.cat([~y_pred, y_pred], dim=1)
            y_true_one_hot = torch.cat([~y_true, y_true], dim=1)
        else:
            raise ValueError("y_pred must be a single channel tensor")
        # Compute the HD95 using MONAI metric
        self.monai_hd95(y_pred_one_hot, y_true_one_hot)

    def compute(self):
        # Return the computed HD95 value from MONAI
        return self.monai_hd95.aggregate()

    def reset(self):
        # Reset the MONAI metric
        self.monai_hd95.reset()


class HD95Metric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Initialize tensor to store individual HD95 scores
        self.add_state(
            "hd95_scores",
            default=torch.tensor([]).to(self.device),
            dist_reduce_fx="cat",
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor
    ):  # input shape of (batch_size, height, width)
        # Convert tensors to binary format
        if preds.dim() == 4:  # (batch_size, channel , height, width)
            preds = preds.squeeze(1)  # (batch_size, height, width)
            target = target.squeeze(1)  # (batch_size, height, width)

        preds_binary = (F.sigmoid(preds) > 0.5).type(torch.int8)
        targets_binary = target.type(torch.int8)
        # Compute sums for the entire batch
        preds_sums = preds_binary.sum(dim=[-1, -2])
        targets_sums = targets_binary.sum(dim=[-1, -2])

        # Determine non-empty mask indices
        non_empty_indices = (preds_sums > 0) & (targets_sums > 0)

        # Placeholder for HD95 scores for the batch
        batch_hd95_scores = torch.zeros(preds.shape[0], dtype=torch.float32).to(
            self.device
        )

        # Compute hd95 only for non-empty masks
        non_empty_tensor = torch.nonzero(non_empty_indices).squeeze()
        non_empty_idx_list = (
            non_empty_tensor.unsqueeze(0)
            if non_empty_tensor.dim() == 0
            else non_empty_tensor
        )

        for idx in non_empty_idx_list:
            hd95_val = hd95(
                preds_binary[idx].cpu().numpy(), targets_binary[idx].cpu().numpy()
            )
            batch_hd95_scores[idx] = hd95_val

        # Batched concatenation to HD95 scores tensor
        self.hd95_scores = torch.cat([self.hd95_scores, batch_hd95_scores], dim=0)

    def compute(self):
        # Return mean HD95 over batch
        return self.hd95_scores.mean()

    # Override the reset method (if necessary)
    def reset(self):
        self.hd95_scores = torch.tensor([]).to(self.device)


def get_binary_metrics(mode="tr", *args, **kwargs):
    if mode not in ["tr", "te", "vl"]:
        raise ValueError("mode must be in ['tr','te','vl']")
    if mode == "te":
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Accuracy(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Dice(multiclass=False, average="samples"),
                torchmetrics.Precision(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Specificity(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Recall(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                # IoU
                torchmetrics.JaccardIndex(task="binary", validate_args=False),
                MONAIHausdorffDistance(),  # only use it for test set
            ],
            prefix="metrics/",
        )
    else:
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Accuracy(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Dice(multiclass=False, average="samples"),
                torchmetrics.Precision(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Specificity(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                torchmetrics.Recall(
                    task="binary", multidim_average="samplewise", validate_args=False
                ),
                # IoU
                torchmetrics.JaccardIndex(task="binary", validate_args=False),
                # torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
            ],
            prefix="metrics/",
        )
    return metrics
    # test_metrics
    # test_metrics = metrics.clone(prefix="").to(device)
    # return test_metrics


def get_binary_metrics_detection(*args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(
                task="binary", multidim_average="samplewise", validate_args=False
            ),
            torchmetrics.Accuracy(
                task="binary", multidim_average="samplewise", validate_args=False
            ),
            # torchmetrics.Dice(multiclass=False, average="samples"),
            torchmetrics.Precision(
                task="binary", multidim_average="samplewise", validate_args=False
            ),
            torchmetrics.Specificity(
                task="binary", multidim_average="samplewise", validate_args=False
            ),
            torchmetrics.Recall(
                task="binary", multidim_average="samplewise", validate_args=False
            ),
            torchmetrics.AUROC(task="binary", validate_args=False),
            torchmetrics.AveragePrecision(task="binary", validate_args=False),
            # IoU
            # torchmetrics.JaccardIndex(task="binary", validate_args=False),
            # torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
        ],
        prefix="metrics/",
    )
    return metrics
