import math
from abc import ABC
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
import torch.nn.functional as F
from torchmetrics import MetricCollection
import torchmetrics.classification as tm_cls 
from typing import Any, Literal, Optional, Tuple # Ensure Tuple is imported

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding, as proposed in "Attention Is All You Need"
    and used in DDPMs.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class ConvBlock(nn.Module):
    """Basic convolutional block with GroupNorm and SiLU activation."""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU() # Swish-like activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResnetBlock(nn.Module):
    """
    Residual block with time embedding.
    """
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )

        self.block1 = ConvBlock(in_channels, out_channels, groups=groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=groups)
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.mlp is not None and time_emb is not None:
            time_emb_out = self.mlp(time_emb)
            # Add time embedding, needs to be reshaped: (B, C) -> (B, C, 1, 1) for broadcasting
            h = h + time_emb_out.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)

class DownBlock(nn.Module):
    """Downsampling block for UNet."""
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.res_block = ResnetBlock(in_channels, out_channels, time_emb_dim=time_emb_dim, groups=groups)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) # Strided conv for downsampling

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        x_down = self.downsample(x)
        return x, x_down # Return skip connection and downsampled output

class UpBlock(nn.Module):
    """Upsampling block for UNet."""
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        # The input channels to the ResNet block will be in_channels (from previous up_block) + skip_channels (from corresponding down_block)
        self.res_block = ResnetBlock(in_channels + skip_channels, out_channels, time_emb_dim=time_emb_dim, groups=groups)
        # Transposed conv for upsampling. Output_padding helps match sizes if stride leads to ambiguity.
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip_x, time_emb):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1) # Concatenate skip connection
        x = self.res_block(x, time_emb)
        return x


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class UNetConditional(nn.Module):
    """
    Conditional UNet architecture for DDPM.
    It takes a noisy image, a timestep, and conditioning data as input.
    The conditioning data is concatenated to the noisy image along the channel dimension.
    """
    def __init__(
        self,
        image_size, # e.g., 64 for 64x64 image
        in_target_channels=1, # For target mask (e.g., 1 for grayscale)
        in_condition_channels=24, # For conditioning environmental data
        model_channels=64, # Base number of channels in the model
        out_channels=1, # Output channels (usually same as in_target_channels for noise prediction)
        num_res_blocks=2, # Number of residual blocks per down/up stage
        channel_mult=(1, 2, 4, 8), # Channel multipliers for each resolution
        time_emb_dim_mult=4, # Multiplier for time embedding dimension relative to model_channels
        groups=8, # GroupNorm groups
    ):
        super().__init__()

        self.channel_mult = channel_mult 
        self.num_res_blocks = num_res_blocks 
        self.image_size = image_size
        self.in_target_channels = in_target_channels
        self.in_condition_channels = in_condition_channels
        total_in_channels = in_target_channels + in_condition_channels # Combined input

        # Time embedding
        time_emb_dim = model_channels * time_emb_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(total_in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        num_resolutions = len(channel_mult)

        for i in range(num_resolutions):
            out_ch = model_channels * channel_mult[i]
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlock(current_channels, out_ch, time_emb_dim=time_emb_dim, groups=groups)
                )
                current_channels = out_ch
            if i != num_resolutions - 1: # Don't add downsample at the last level
                self.down_blocks.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=4, stride=2, padding=1) # Downsample
                )

        # Bottleneck
        self.mid_block1 = ResnetBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, groups=groups)
        self.mid_block2 = ResnetBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, groups=groups)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        current_channels = model_channels * channel_mult[-1] # Start from bottleneck channels
        for i in reversed(range(num_resolutions)):
            expected_skip_channels = model_channels * channel_mult[i]
            # Upsample layer if not the first upsampling stage (i.e. if we are not at bottleneck resolution)
            if i != num_resolutions -1 : 
                 self.up_blocks.append(
                    nn.ConvTranspose2d(current_channels, expected_skip_channels, kernel_size=4, stride=2, padding=1)
                 )
                 current_channels = expected_skip_channels

            # ResNet blocks for this resolution
            # The first ResNet block in an upsampling stage takes current_channels + skip_channels
            # Subsequent ones take expected_skip_channels
            for j in range(num_res_blocks +1): # +1 for the block that receives the concatenated features
                if j == 0: # Block that receives concatenated features
                    block_in_channels = current_channels + expected_skip_channels
                else: # Subsequent blocks
                    block_in_channels = expected_skip_channels

                self.up_blocks.append(
                    ResnetBlock(block_in_channels, expected_skip_channels, time_emb_dim=time_emb_dim, groups=groups)
                )
            current_channels = expected_skip_channels


        # Final layer
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=1) 

    def forward(self, x_t, time, context):
        """
        Args:
            x_t (torch.Tensor): Noisy target image (B, C, H, W)
            time (torch.Tensor): Timesteps (B,)
            context (torch.Tensor): Conditioning data (B, C, H, W)
        """
        # ---> ADD THIS DEFENSIVE FIX <---
        # If x_t is 3D and context is 4D, it means x_t is likely missing a dimension.
        # Let's expand it to match the context.
        if x_t.ndim == 3 and context.ndim == 4:
            # Assuming x_t is (B, C, H) and needs a W dimension.
            x_t = x_t.unsqueeze(-1)  # Shape becomes (B, C, H, 1)
            # Expand the new dimension to match the context's width.
            x_t = x_t.expand(-1, -1, -1, context.shape[3])
        # -----------------------------

        # Now, both tensors should be 4D. The original check can proceed.
        if x_t.shape[2:] != context.shape[2:]:
            context = F.interpolate(context, size=x_t.shape[2:], mode='bilinear', align_corners=False)

        nn_input = torch.cat((x_t, context), dim=1)

        # 2. Compute time embedding
        t_emb = self.time_mlp(time) 

        # 3. Initial convolution
        h = self.init_conv(nn_input) 
        
        # Skip connections
        skips = [h] 

        # 4. Downsampling path
        block_idx = 0
        num_resolutions = len(self.channel_mult) # Corrected loop range
        for i in range(num_resolutions):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx +=1
            if i < num_resolutions -1 : # If not the last resolution
                h = self.down_blocks[block_idx](h) # Downsample conv
                skips.append(h) 
                block_idx +=1


        # 5. Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # 6. Upsampling path
        block_idx = 0
        for i in reversed(range(num_resolutions)):
            if i < num_resolutions -1 : 
                h = self.up_blocks[block_idx](h) # Upsample conv
                block_idx +=1

            # Concatenate with skip connection. Skips are stored in reverse order of use
            for j in range(self.num_res_blocks +1):
                skip_h = skips.pop()
                if h.size(2) != skip_h.size(2) or h.size(3) != skip_h.size(3): # Ensure spatial dims match for concat
                     skip_h = F.interpolate(skip_h, size=h.shape[2:], mode='bilinear', align_corners=False)

                if j == 0: # First block in upsampling stage: concatenate skip connection
                    h = self.up_blocks[block_idx](torch.cat((h, skip_h), dim=1), t_emb)
                else: # Subsequent blocks already assume the correct number of input channels
                    h = self.up_blocks[block_idx](h, t_emb)
                block_idx += 1

        # 7. Final layer
        output = self.final_conv(h)
        return output

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for DDPM.
    Handles noise scheduling, forward diffusion (q_sample), and reverse denoising (p_sample).
    """
    # In src/models/ddpmModel.py, inside the GaussianDiffusion class

    def __init__(self, model, image_size, timesteps=20, beta_schedule_type='linear',
                target_channels=1):
        super().__init__()
        self.model = model  # The UNet model
        self.image_size = image_size
        self.target_channels = target_channels  # Channels of the image being diffused
        self.timesteps = timesteps

        # --- Corrected Section using register_buffer ---

        # 1. Define beta schedule
        if beta_schedule_type == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule_type == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule type: {beta_schedule_type}")

        # 2. Register all schedule-related tensors as buffers
        self.register_buffer('betas', betas)
        
        alphas = 1. - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, axis=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Clip variance to avoid 0
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))

    def _extract(self, a, t, x_shape):
        """Extracts values from a at specified timesteps t and reshapes for broadcasting."""
        batch_size = t.shape[0]
        # Ensure 't' is on the same device as 'a' for gather operation
        out = a.gather(-1, t.to(a.device)) # Explicitly move t to a's device
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Samples x_t by adding noise to x_0.
        x_start: Original clean image (B, C, H, W)
        t: Timestep (B,)
        noise: Optional noise tensor; if None, generated from N(0,1)
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, context, noise=None, loss_type="l2"):
        """
        Calculates the loss for training the DDPM.
        Predicts the noise added to x_start at timestep t, conditioned on context.
        x_start: Original clean image (B, C, H, W)
        t: Timestep (B,)
        context: Conditioning data (B, C_cond, H, W)
        noise: The noise that was added (if None, generate new noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_noise = self.model(x_noisy, t, context) # UNet predicts the noise

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_mean_variance(self, x, t, context):
        batch_size = x.shape[0]

        # Get values from pre-computed schedules
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        
        # Predict noise using the model
        predicted_noise = self.model(x, t, context) 

        # Calculate mean and variance
        model_mean = (x - betas_t * predicted_noise) / torch.sqrt(1. - alphas_cumprod_t)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        return model_mean, posterior_variance_t, posterior_log_variance_clipped_t

    @torch.no_grad()
    def p_sample(self, x, t, context, t_index):
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x=x, t=t, context=context)
        noise = torch.randn_like(x)
        # No noise if t == 0
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, context, batch_size=1, channels=1):

        # Initial noise for sampling
        img = torch.randn((batch_size, channels, self.image_size, self.image_size), device=context.device)
        
        intermediate_steps = []
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, context, i) # Pass t_index
            intermediate_steps.append(img.cpu()) # Store intermediate steps on CPU to save GPU memory
        return img.cpu(), intermediate_steps # Return final image and intermediate steps

class DDPMLightning(pl.LightningModule, ABC):
    """
    This class implements a Denoising Diffusion Probabilistic Model (DDPM)
    adapted to the project's BaseModel structure. It handles training, validation,
    and testing for generating fire spread masks conditioned on environmental data.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["L1", "L2", "Huber"], # For DDPM, this indicates the noise loss type
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
            flatten_temporal_dimension (bool): Flag for compatibility, logic is handled in steps.
            pos_class_weight (float): Unused in DDPM, but kept for compatibility with the script.
            loss_function (str): The type of loss to use for comparing predicted noise to actual noise (e.g., "L2").
            unet_params (Dict): Dictionary of parameters for the UNetConditional model.
            diffusion_params (Dict): Dictionary of parameters for the GaussianDiffusion process.
            optimizer_cfg (Dict): Configuration for the optimizer.
            metrics_cfg (Dict): Configuration for evaluation metrics.
        """
        super().__init__(*args, **kwargs)
        # Save all hyperparameters. This makes them accessible via self.hparams
        self.save_hyperparameters()

        # --- Initialize the core DDPM components ---
        # Update the U-Net to accept the correct number of conditioning channels
        unet_params["in_condition_channels"] = n_channels
        self.unet = UNetConditional(**unet_params)
        self.diffusion = GaussianDiffusion(model=self.unet, **diffusion_params)

        # --- Setup metrics for evaluation ---
        # Note: The DDPM loss is not a classification loss, so we only track the noise prediction loss during training/validation.
        self.train_metrics = torchmetrics.MetricCollection({"train_loss": torchmetrics.MeanMetric()})
        self.val_metrics = self.train_metrics.clone(prefix="val_")

        # For testing, we generate an image and compare it to the ground truth using standard classification metrics.
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

    def _common_step(self, batch: tuple):
        """Handles the core logic for a single training or validation step."""
        conditions, targets = batch

        # --- Data Preparation ---
        # This logic handles the `flatten_temporal_dimension` concept by reshaping the input
        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)
        if targets.ndim == 5:
            targets = targets[:, -1, :, :, :]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[1] > 1:
            targets = targets[:, 0:1, :, :]

        # --- DDPM Loss Calculation ---
        targets_binary = (targets == 255.0).float()
        x_start = (targets_binary * 2) - 1 # Scale target to [-1, 1]
        t = torch.randint(0, self.diffusion.timesteps, (x_start.shape[0],), device=self.device).long()
        loss = self.diffusion.p_losses(x_start=x_start, t=t, context=conditions, loss_type=self.hparams.loss_function.lower())

        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        """Compute and log the training loss."""
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """Compute and log the validation loss."""
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int):
        """Generate a full image from noise and evaluate it against the ground truth."""
        conditions, targets = batch

    # --- START OF NEW DIAGNOSTIC CODE ---
    # We will only run this for the very first batch (batch_idx == 0)
        if batch_idx == 0:
            print("--- Running Data Sanity Check ---")
            
            # 1. Save the last channel of the conditioning data (e.g., the active fire mask)
            # We need to process it a bit to make it a viewable image
            input_fire_mask = conditions[0, -1, :, :].cpu().numpy() # Take first item in batch, last channel
            plt.imsave("debug_input_fire_mask.png", input_fire_mask, cmap='gray')
            print("Saved 'debug_input_fire_mask.png' to disk.")

            # 2. Save the ground truth target image
            # We need to process it to ensure it's in the right shape and on the CPU
            target_image = targets[0].cpu() # Take the first item in the batch
            if target_image.ndim == 3:
                target_image = target_image.squeeze(0) # Remove channel dimension if it exists
            plt.imsave("debug_target_image.png", target_image.numpy(), cmap='gray')
            print("Saved 'debug_target_image.png' to disk.")
            print("--- Data Sanity Check Complete ---")
    # --- END OF NEW DIAGNOSTIC CODE ---

        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)

        generated_samples, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels']
        )
        
        predicted_probs = (generated_samples + 1) / 2.0
        predicted_probs = torch.clamp(predicted_probs, 0.0, 1.0).to(targets.device)

        # Reshape and resize targets to match predictions
        if targets.ndim == 5:
            targets = targets[:, -1, :, :, :]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[1] > 1:
            targets = targets[:, 0:1, :, :]
        if targets.shape[-2:] != predicted_probs.shape[-2:]:
            targets = F.interpolate(targets.float(), size=predicted_probs.shape[-2:], mode='nearest')

        targets_binary = (targets == 255.0).int()

        self.test_metrics.update(predicted_probs.flatten(), targets_binary.flatten())
        self.test_conf_mat.update(predicted_probs.flatten(), targets_binary.flatten())
        self.test_pr_curve.update(predicted_probs.flatten(), targets_binary.flatten())

        """
        print(self.test_metrics["test_AP"])
        print(self.test_metrics["test_F1"])
        print(self.test_metrics["test_Precision"])
        print(self.test_metrics["test_Recall"])
        print(self.test_metrics["test_IOU"])
        """

    def on_test_epoch_end(self):
        """Log the final test metrics, confusion matrix, and PR curve."""
        self.log_dict(self.test_metrics.compute())
        conf_mat = self.test_conf_mat.compute().cpu().numpy()
        wandb_table = wandb.Table(data=conf_mat, columns=["Predicted Background", "Predicted Fire"])
        wandb.log({"Test Confusion Matrix": wandb_table})
        fig, ax = self.test_pr_curve.plot(score=True)
        wandb.log({"Test PR Curve": wandb.Image(fig)})
        plt.close()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Generate raw image predictions for new data."""
        if len(batch) == 2:
            conditions, _ = batch
        else:
            conditions = batch[0]
        if conditions.ndim == 5:
            b, t, c, h, w = conditions.shape
            conditions = conditions.view(b, t * c, h, w)
        generated_samples, _ = self.diffusion.sample(
            context=conditions,
            batch_size=conditions.shape[0],
            channels=self.hparams.unet_params['out_channels']
        )
        predicted_probs = (generated_samples + 1) / 2.0
        return torch.clamp(predicted_probs, 0.0, 1.0)

    def configure_optimizers(self):
        """Configure the optimizer for the U-Net model based on the provided config."""
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.hparams.optimizer_cfg['init_args']['lr'],
            weight_decay=self.hparams.optimizer_cfg['init_args']['weight_decay']
        )
        return optimizer