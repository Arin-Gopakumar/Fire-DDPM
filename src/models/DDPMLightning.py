import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics.classification as tm_cls 
import pytorch_lightning as pl 
from typing import Any, Literal, Optional, Tuple # Ensure Tuple is imported

# Assuming UNetConditional and GaussianDiffusion are in src/models/ddpmModel.py
from models.ddpmModel import UNetConditional, GaussianDiffusion

class DDPMLightning(pl.LightningModule):
    """
    PyTorch Lightning Module for the Conditional Denoising Diffusion Probabilistic Model (DDPM).
    This module wraps the UNetConditional and GaussianDiffusion components.
    It is adapted to fit the existing codebase's structure and LightningCLI.
    """
    def __init__(
        self, 
        # All non-default arguments first
        n_channels: int, 
        flatten_temporal_dimension: bool, 
        pos_class_weight: float, 
        loss_function: str, # Literal type converted to str for simplicity
        unet_params: dict, 
        diffusion_params: dict,
        optimizer_cfg: dict, 
        metrics_cfg: dict,   
        
        # All arguments with default values next
        use_doy: bool = False, 
        required_img_size: Optional[Tuple[int, int]] = None, 
        
        **kwargs # Catches any other unexpected keyword arguments passed by LightningCLI
    ):
        super().__init__() # Call LightningModule's init
        
        # Save hyperparameters for easy logging and checkpointing by Lightning
        self.save_hyperparameters(logger=True) 

        # Instantiate the UNetConditional model
        self.unet = UNetConditional(**unet_params)

        # Instantiate the GaussianDiffusion process, passing the UNet model
        self.diffusion = GaussianDiffusion(model=self.unet, **diffusion_params)

        # Initialize metrics for different stages (train, val, test)
        self.train_metrics = self._setup_metrics("train", metrics_cfg)
        self.val_metrics = self._setup_metrics("val", metrics_cfg)
        self.test_metrics = self._setup_metrics("test", metrics_cfg)

    def _setup_metrics(self, stage: str, metrics_cfg: dict):
        """Helper to set up MetricCollection for a given stage."""
        return MetricCollection({
            'AP': tm_cls.BinaryAveragePrecision(),
            'Precision': tm_cls.BinaryPrecision(),
            'Recall': tm_cls.BinaryRecall(),
            'IoU': tm_cls.BinaryJaccardIndex(),
            #'Dice': tm_cls.BinaryDice()
        }, prefix=f"{stage}_")

    def forward(self, x_t, t, context):
        """
        Forward pass of the UNet model. Used internally by GaussianDiffusion.
        """
        return self.unet(x_t, t, context)

    def _common_step(self, batch: tuple, stage: str):
        """
        ***********Question: How should get_pred_and_gt receive and unpack the batch from the DataLoader? Should it expect (conditions, targets, ids) as a tuple, or continue to expect (x, y) or (x, y, doys)?************
        Common logic for training, validation, and test steps.
        Unpacks batch as (conditions, targets, ids).
        """

        conditions, targets = batch 
        conditions = conditions.to(self.device)
        targets = targets.to(self.device)

        # CRITICAL: Binarize targets for the diffusion model and loss calculation
        # prepare_data.py outputs targets where 255.0 is "fire" (positive class) and 0.0 is "no fire" (negative class).
        # So, targets_binary will be 1.0 for fire, 0.0 for no fire.
        targets_binary = (targets == 255.0).float() 

        # Scale targets to [-1, 1] range for the diffusion model (x_start)
        x_start = (targets_binary * 2) - 1 

        # Generate random timesteps for the diffusion process
        t = torch.randint(0, self.diffusion.timesteps, (targets.shape[0],), device=self.device).long()

        # Calculate the diffusion loss (e.g., L2 loss between predicted noise and true noise)
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type="l2")

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        loss = self._common_step(batch, "train")
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        loss = self._common_step(batch, "val")
        return loss

    def test_step(self, batch: tuple, batch_idx: int):
        conditions, targets, _ = batch 
        conditions = conditions.to(self.device)
        targets = targets.to(self.device)
        
        generated_samples_scaled, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels'] 
        )
        predicted_probs = (generated_samples_scaled + 1) / 2.0
        predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0)

        targets_binary = (targets == 255.0).int()

        self.test_metrics.update(predicted_probs.flatten(), targets_binary.flatten())
        
        x_start = (targets_binary * 2) - 1 
        t = torch.randint(0, self.diffusion.timesteps, (targets.shape[0],), device=self.device).long()
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type="l2")
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True) 
        
        return loss

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_cfg['class_path'].split('.')[-1])
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer_cfg['init_args'])
        return optimizer

