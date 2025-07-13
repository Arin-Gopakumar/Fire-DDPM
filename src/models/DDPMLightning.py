import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics.classification as tm_cls 
import pytorch_lightning as pl 

# Removed: from models.BaseModel import BaseModel # No longer inheriting from BaseModel

# Assuming UNetConditional and GaussianDiffusion are in src/models/ddpmModel.py
from models.ddpmModel import UNetConditional, GaussianDiffusion

# CRITICAL FIX: Inherit directly from pl.LightningModule
class DDPMLightning(pl.LightningModule):
    """
    PyTorch Lightning Module for the Conditional Denoising Diffusion Probabilistic Model (DDPM).
    This module wraps the UNetConditional and GaussianDiffusion components.
    """
    def __init__(self, unet_params: dict, diffusion_params: dict, optimizer_cfg: dict, loss_cfg: dict, metrics_cfg: dict):
        """
        Initializes the DDPMLightning module.

        Args:
            unet_params (dict): Parameters for the UNetConditional model.
            diffusion_params (dict): Parameters for the GaussianDiffusion process.
            optimizer_cfg (dict): Configuration for the optimizer.
            loss_cfg (dict): Configuration for the loss function (will be a dummy as p_losses handles it).
            metrics_cfg (dict): Configuration for evaluation metrics.
        """
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

        # A dummy loss function is needed by LightningModule if not explicitly returned by step methods
        # However, p_losses computes the actual loss.
        # This is typically not needed if step methods return the loss directly.
        # self.loss_fn = nn.Identity() 

    def _setup_metrics(self, stage: str, metrics_cfg: dict):
        """Helper to set up MetricCollection for a given stage."""
        return MetricCollection({
            'AP': tm_cls.BinaryAveragePrecision(),
            'Precision': tm_cls.BinaryPrecision(),
            'Recall': tm_cls.BinaryRecall(),
            'IoU': tm_cls.BinaryJaccardIndex(),
            'Dice': tm_cls.BinaryDice()
        }, prefix=f"{stage}_")

    def forward(self, x_t, t, context):
        """
        Forward pass of the UNet model. Used internally by GaussianDiffusion.
        """
        return self.unet(x_t, t, context)

    def _common_step(self, batch: dict, stage: str):
        """
        Common logic for training, validation, and test steps.
        """
        conditions = batch["condition"] 
        targets = batch["target"]       

        # Binarize targets for the diffusion model and loss calculation
        targets_binary = (targets == 255.0).float() 

        # Scale targets to [-1, 1] range for the diffusion model (x_start)
        x_start = (targets_binary * 2) - 1 

        # Generate random timesteps for the diffusion process
        t = torch.randint(0, self.diffusion.timesteps, (targets.shape[0],), device=self.device).long()

        # Calculate the diffusion loss (e.g., L2 loss between predicted noise and true noise)
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type="l2")

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        loss = self._common_step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss = self._common_step(batch, "val")
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        conditions = batch["condition"]
        targets = batch["target"]
        
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
        pass # No metrics computed in val_step currently

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_epoch=True, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        # Instantiate optimizer from config
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_cfg['class_path'].split('.')[-1])
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer_cfg['init_args'])
        return optimizer

