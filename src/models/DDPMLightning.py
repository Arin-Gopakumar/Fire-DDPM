import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import AveragePrecision, Precision, Recall, JaccardIndex, Dice
import pytorch_lightning as pl # Ensure pytorch_lightning is imported if BaseModel uses it directly

# Assuming BaseModel is in src/models/BaseModel.py
from models.BaseModel import BaseModel 
# Assuming UNetConditional and GaussianDiffusion are in src/models/ddpmModel.py
from models.ddpmModel import UNetConditional, GaussianDiffusion

class DDPMLightning(BaseModel):
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
        # BaseModel expects optimizer, loss, and metrics configs.
        # For DDPM, p_losses calculates the loss, so loss_cfg can be a dummy.
        super().__init__(optimizer=optimizer_cfg, loss=loss_cfg, metrics=metrics_cfg)
        
        # Save hyperparameters for easy logging and checkpointing by Lightning
        self.save_hyperparameters(logger=True)

        # Instantiate the UNetConditional model
        self.unet = UNetConditional(**unet_params)

        # Instantiate the GaussianDiffusion process, passing the UNet model
        # Note: diffusion_params should NOT contain 'model' as it's passed here.
        # Ensure 'image_size', 'timesteps', 'beta_schedule_type', 'target_channels' are in diffusion_params.
        self.diffusion = GaussianDiffusion(model=self.unet, **diffusion_params)

        # Initialize metrics for different stages (train, val, test)
        # We'll use MetricCollection for convenience
        self.train_metrics = self._setup_metrics("train", metrics_cfg)
        self.val_metrics = self._setup_metrics("val", metrics_cfg)
        self.test_metrics = self._setup_metrics("test", metrics_cfg)

        # A dummy loss function is needed by BaseModel, even if p_losses computes the actual loss
        self.loss_fn = nn.Identity() 

    def _setup_metrics(self, stage: str, metrics_cfg: dict):
        """Helper to set up MetricCollection for a given stage."""
        return MetricCollection({
            'AP': AveragePrecision(task="binary"),
            'Precision': Precision(task="binary"),
            'Recall': Recall(task="binary"),
            'IoU': JaccardIndex(task="binary"),
            'Dice': Dice(task="binary")
        }, prefix=f"{stage}_").to(self.device if hasattr(self, 'device') else 'cpu') # Ensure metrics are on correct device

    def forward(self, x_t, t, context):
        """
        Forward pass of the UNet model. Used internally by GaussianDiffusion.
        This is the abstract method from BaseModel.
        """
        return self.unet(x_t, t, context)

    def _common_step(self, batch: dict, stage: str):
        """
        Common logic for training, validation, and test steps.
        """
        conditions = batch["condition"] # (B, C_in, H, W)
        targets = batch["target"]       # (B, 1, H, W) - values are 0.0 or 255.0

        # CRITICAL: Binarize targets for the diffusion model and loss calculation
        # prepare_data.py outputs targets where 255.0 is "fire" (positive class) and 0.0 is "no fire" (negative class).
        # So, targets_binary will be 1.0 for fire, 0.0 for no fire.
        targets_binary = (targets == 255.0).float() # This is the 0.0/1.0 binary target

        # Scale targets to [-1, 1] range for the diffusion model (x_start)
        x_start = (targets_binary * 2) - 1 

        # Generate random timesteps for the diffusion process
        t = torch.randint(0, self.diffusion.timesteps, (targets.shape[0],), device=self.device).long()

        # Calculate the diffusion loss (e.g., L2 loss between predicted noise and true noise)
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type="l2")

        # For metrics, we need the model's predicted probabilities (0-1) and the binary target (0/1)
        # To get predicted probabilities, we can run a sampling step.
        # However, for metric calculation within training/validation, it's more common to
        # use the model's output (predicted noise) to derive a 'denoised' image.
        # For simplicity and consistency with evaluate.py, we'll assume the model's
        # denoised output (if it were to predict x_0 directly) would be scaled to [0,1].
        # Since p_losses is what's optimized, we'll log its value.
        # For evaluation metrics (AP, IoU etc.), we need predicted probabilities.
        # This is typically done by running a full sampling process or a denoising step.
        # For now, we'll just log the training loss and update metrics on validation/test end.
        
        # Log the loss
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        loss = self._common_step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss = self._common_step(batch, "val")
        # For validation, we also compute metrics.
        # To get predictions for metrics, we run a sampling step.
        # Note: Running full sampling in validation_step can be slow.
        # For a full evaluation, evaluate.py is better.
        # Here, we'll just use the loss for early stopping/best model saving.
        # If you need metrics per validation step, you'd run self.diffusion.sample
        # or a denoising step here and update self.val_metrics.
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        loss = self._common_step(batch, "test")
        # For testing, compute and update metrics
        conditions = batch["condition"]
        targets = batch["target"]
        
        # Run a full sampling process to get predicted probabilities
        # This can be computationally intensive for every test step.
        # Consider running inference only on a subset or in a separate script (like evaluate.py).
        # For now, we'll use the final denoised output from the model for a quick metric update.
        
        # Get the model's predicted x_0 (denoised image)
        # This requires running the reverse diffusion process.
        # For simplicity in test_step, we'll assume a single denoising pass from noise.
        # A more robust approach for metrics would be to run self.diffusion.sample here,
        # but that's typically done in evaluate.py.
        
        # For test metrics, we need predicted probabilities (0-1)
        # Let's run the full sampling process here as it's the most accurate way to get predictions
        # for metrics, similar to evaluate.py.
        generated_samples_scaled, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels'] # Use out_channels of UNet
        )
        predicted_probs = (generated_samples_scaled + 1) / 2.0
        predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0)

        # Binarize targets for metrics (consistent with _common_step)
        targets_binary = (targets == 255.0).int()

        self.test_metrics.update(predicted_probs.flatten(), targets_binary.flatten())
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True) # Log loss for epoch end
        return loss

    def on_validation_epoch_end(self):
        # Compute and log validation metrics
        # If metrics were updated in validation_step, compute them here.
        # Since we only logged loss in validation_step for simplicity, we'll just log loss.
        # If you add metric updates in validation_step, uncomment and use self.val_metrics.compute()
        # self.log_dict(self.val_metrics.compute(), on_epoch=True, logger=True)
        pass # No metrics computed in val_step currently

    def on_test_epoch_end(self):
        # Compute and log test metrics
        self.log_dict(self.test_metrics.compute(), on_epoch=True, logger=True)
        # After computing, reset metrics for the next test run (if any)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Configures the optimizer based on the optimizer_cfg.
        """
        # Instantiate optimizer from config
        optimizer_class = getattr(torch.optim, self.hparams.optimizer['class_path'].split('.')[-1])
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer['init_args'])
        return optimizer

