import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from Loss.dice import DiceLoss


class TotalLoss(nn.Module):
    """
    Final loss:
    1. Main segmentation loss:
       SoftCrossEntropyLoss + DiceLoss

    2. Edge auxiliary loss:
       Binary DiceLoss for boundary prediction

    Supported outputs:
    - logits = logit_main
    - logits = (logit_main, logit_edge)
    """

    def __init__(self, ignore_index=255, edge_weight=0.5):
        super().__init__()

        self.ignore_index = ignore_index
        self.edge_weight = edge_weight

        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(
                smooth_factor=0.05,
                ignore_index=ignore_index
            ),
            DiceLoss(
                smooth=0.05,
                ignore_index=ignore_index
            ),
            1.0,
            1.0
        )

        self.edge_loss = DiceLoss(
            mode='binary',
            from_logits=True
        )

    def get_boundary(self, labels):
        """
        Generate binary boundary map from segmentation labels.
        labels: [N, H, W]
        return: [N, 1, H, W]
        """

        device = labels.device

        # avoid ignore_index affecting boundary generation
        labels_for_edge = labels.clone()
        labels_for_edge[labels_for_edge == self.ignore_index] = 0

        laplacian_kernel = torch.tensor(
            [-1, -1, -1,
             -1,  8, -1,
             -1, -1, -1],
            dtype=torch.float32,
            device=device
        ).reshape(1, 1, 3, 3)

        labels_for_edge = labels_for_edge.unsqueeze(1).float()

        boundary = F.conv2d(
            labels_for_edge,
            laplacian_kernel,
            padding=1
        )

        boundary = boundary.clamp(min=0)
        boundary[boundary >= 0.1] = 1
        boundary[boundary < 0.1] = 0

        return boundary

    def forward(self, logits, labels):
        """
        logits:
        - Tensor: main segmentation prediction
        - Tuple/List:
            [0] logit_main: segmentation prediction
            [1] logit_edge: boundary prediction
            [2] edge_2: optional second boundary prediction

        labels:
        - [N, H, W]
        """

        if isinstance(logits, (tuple, list)):
            logit_main = logits[0]

            if labels.device != logit_main.device:
                labels = labels.to(logit_main.device)

            loss_main = self.main_loss(logit_main, labels)

            if self.training and len(logits) >= 2:
                boundary = self.get_boundary(labels)

                edge_losses = []

                for edge_logit in logits[1:]:
                    if edge_logit.device != boundary.device:
                        edge_logit = edge_logit.to(boundary.device)

                    edge_losses.append(
                        self.edge_loss(edge_logit, boundary)
                    )

                loss_edge = sum(edge_losses) / len(edge_losses)

                loss = (
                    (1.0 - self.edge_weight) * loss_main
                    + self.edge_weight * loss_edge
                )

            else:
                loss = loss_main

        else:
            if labels.device != logits.device:
                labels = labels.to(logits.device)

            loss = self.main_loss(logits, labels)

        return loss