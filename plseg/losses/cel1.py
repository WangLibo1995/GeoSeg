import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

BINARY_MODE: str = "binary"

MULTICLASS_MODE: str = "multiclass"

MULTILABEL_MODE: str = "multilabel"


EPS = 1e-10


logger = logging.getLogger(__name__)


def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask


def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bcwh->bc", valid_mask)
        else:
            x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3]

    region_proportion = (torch.einsum("bcwh->bc", x) + EPS) / (cardinality + EPS)

    return region_proportion


class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, mode: str = MULTICLASS_MODE,
                 alpha: float = 0.1,
                 factor: float = 5.,
                 step_size: int = 0,
                 max_alpha: float = 100.,
                 temp: float = 1.,
                 ignore_index: int = 255,
                 background_index: int = -1,
                 weight: Optional[torch.Tensor] = None) -> None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        if self.mode == MULTICLASS_MODE:
            loss = F.cross_entropy(
                inputs, labels, weight=self.weight, ignore_index=self.ignore_index)
        else:
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            logger.info(
                "CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}".format(curr_alpha, self.alpha)
            )

    def get_gt_proportion(self, mode: str,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 255):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            bin_labels = labels
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: str,
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        loss_ce = self.cross_entropy(inputs, labels)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()

        loss = loss_ce + self.alpha * loss_reg

        return loss


class CrossEntropyWithKL(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * KL(gt_region || prob_region)
    """
    def kl_div(self, p : torch.Tensor, q : torch.Tensor) -> torch.Tensor:
        x = p * torch.log(p / q)
        x = torch.einsum("ij->i", x)
        return x

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        loss_ce = self.cross_entropy(inputs, labels)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)

        if self.mode == BINARY_MODE:
            regularizer = (
                self.kl_div(gt_proportion, pred_proportion)
                + self.kl_div(1 - gt_proportion, 1 - pred_proportion)
            ).mean()
        else:
            regularizer = self.kl_div(gt_proportion, pred_proportion).mean()

        loss = loss_ce + self.alpha * regularizer

        return loss