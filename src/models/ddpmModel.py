import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        img = torch.randn((batch_size, channels, self.image_size, self.image_size), device=self.device)
        
        intermediate_steps = []
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, context, i) # Pass t_index
            intermediate_steps.append(img.cpu()) # Store intermediate steps on CPU to save GPU memory
        return img.cpu(), imgs # Return final image and intermediate steps
