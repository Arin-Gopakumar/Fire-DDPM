import math
from abc import ABC
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from models.ddpmModel import GaussianDiffusion, UNetConditional # Assuming your DDPM components are in this file

class DDPMLightning(pl.LightningModule, ABC):
    """
    This class implements a Denoising Diffusion Probabilistic Model (DDPM)
    within the PyTorch Lightning framework. It handles the training, validation,
    and testing loops for generating images (e.g., fire masks) conditioned on
    environmental data.
    """
    def __init__(
        self,
        n_channels: int,
        pos_class_weight: float,
        unet_params: Dict,
        diffusion_params: Dict,
        optimizer_cfg: Dict,
        metrics_cfg: Dict,
        *args: Any,
        **kwargs: Any
    ):
        """
        Args:
            n_channels (int): Number of input channels from the conditioning data.
            pos_class_weight (float): Unused in DDPM, but kept for compatibility.
            unet_params (Dict): Dictionary of parameters for the UNetConditional model.
            diffusion_params (Dict): Dictionary of parameters for the GaussianDiffusion process.
            optimizer_cfg (Dict): Configuration for the optimizer (e.g., AdamW).
            metrics_cfg (Dict): Configuration for evaluation metrics.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # 1. Initialize the core DDPM components
        self.unet = UNetConditional(**unet_params)
        self.diffusion = GaussianDiffusion(model=self.unet, **diffusion_params)

        # 2. Setup metrics for evaluation
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "train_loss": torchmetrics.MeanMetric()
            }
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "test_AP": torchmetrics.AveragePrecision("binary"),
                "test_F1": torchmetrics.F1Score("binary"),
                "test_Precision": torchmetrics.Precision("binary"),
                "test_Recall": torchmetrics.Recall("binary"),
                "test_IOU": torchmetrics.JaccardIndex("binary"),
            }
        )
        self.test_conf_mat = torchmetrics.ConfusionMatrix("binary")
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)


    def _common_step(self, batch: tuple, stage: str):
        """
        Handles the core logic for a single training or validation step.
        """
        conditions, targets = batch

        # --- Ensure data has the correct 4D shape ---
        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)
        if targets.ndim == 5:
            targets = targets[:, -1, :, :, :] # Select last image in time series
        if targets.ndim == 3:
            targets = targets.unsqueeze(1) # Add channel dimension
        if targets.shape[1] > 1:
            targets = targets[:, 0:1, :, :] # Select first channel

        # --- DDPM Logic ---
        # 1. Prepare the target image: binarize and scale to [-1, 1]
        targets_binary = (targets == 255.0).float()
        x_start = (targets_binary * 2) - 1

        # 2. Sample a random timestep
        t = torch.randint(0, self.diffusion.timesteps, (x_start.shape[0],), device=self.device).long()

        # 3. Calculate the diffusion loss (typically L2 a.k.a. MSE)
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type="l2")

        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Compute and log the training loss for the DDPM.
        """
        loss = self._common_step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Compute and log the validation loss.
        """
        loss = self._common_step(batch, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Generate a full image from noise and evaluate it against the ground truth.
        """
        conditions, targets = batch

        # --- Prepare data, same as in _common_step ---
        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)

        # 1. Generate an image from noise via the reverse diffusion process
        generated_samples_scaled, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels']
        )
        
        # 2. Convert the generated image from [-1, 1] to [0, 1] probabilities
        predicted_probs = (generated_samples_scaled + 1) / 2.0
        predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0)
        predicted_probs = predicted_probs.to(targets.device) # Ensure device match

        # --- Prepare targets and resize if necessary ---
        if targets.ndim == 5:
            targets = targets[:, -1, :, :, :]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[1] > 1:
            targets = targets[:, 0:1, :, :]
        if targets.shape[-2:] != predicted_probs.shape[-2:]:
            targets = F.interpolate(targets.float(), size=predicted_probs.shape[-2:], mode='nearest')

        targets_binary = (targets == 255.0).int()

        # 3. Update all test metrics
        self.test_metrics.update(predicted_probs.flatten(), targets_binary.flatten())
        self.test_conf_mat.update(predicted_probs.flatten(), targets_binary.flatten())
        self.test_pr_curve.update(predicted_probs.flatten(), targets_binary.flatten())

    def on_test_epoch_end(self):
        """
        Log the final test metrics, confusion matrix, and PR curve.
        """
        self.log_dict(self.test_metrics.compute())

        conf_mat = self.test_conf_mat.compute().cpu().numpy()
        wandb_table = wandb.Table(data=conf_mat, columns=["Predicted Background", "Predicted Fire"])
        wandb.log({"Test Confusion Matrix": wandb_table})

        fig, ax = self.test_pr_curve.plot(score=True)
        wandb.log({"Test PR Curve": wandb.Image(fig)})
        plt.close()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Generate raw image predictions for new data.
        """
        if len(batch) == 2:
            conditions, _ = batch
        else:
            conditions = batch[0]

        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)

        generated_samples_scaled, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels']
        )
        
        predicted_probs = (generated_samples_scaled + 1) / 2.0
        predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0)
        
        return predicted_probs

    def configure_optimizers(self):
        """
        Configure the optimizer for the U-Net model.
        """
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.hparams.optimizer_cfg['init_args']['lr'],
            weight_decay=self.hparams.optimizer_cfg['init_args']['weight_decay']
        )
        return optimizer