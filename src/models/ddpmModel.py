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

        self.channel_mult = channel_mult #Added this bc gpt told me to
        self.num_res_blocks = num_res_blocks #Added this bc gpt told me to 
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
        for i in reversed(range(num_resolutions)):
            out_ch = model_channels * channel_mult[i]
            for _ in range(num_res_blocks + 1): # +1 because one resblock before upsample, one after for skip
                # Input channels to ResNet block = current_channels (from up) + out_ch (from skip)
                self.up_blocks.append(
                    ResnetBlock(current_channels + out_ch if _ == 0 else out_ch, # first block in stage gets skip
                                out_ch, time_emb_dim=time_emb_dim, groups=groups)
                )
                current_channels = out_ch # This logic is slightly off, fixed below
            if i != 0: # Don't add upsample at the first level (highest res)
                self.up_blocks.append(
                     nn.ConvTranspose2d(current_channels, current_channels // channel_mult[i-1] * channel_mult[i-1] if i > 0 else model_channels,
                                       kernel_size=4, stride=2, padding=1) # Upsample
                )
        # Corrected Upsampling path logic
        self.up_blocks = nn.ModuleList()
        current_channels = model_channels * channel_mult[-1] # Start from bottleneck channels
        for i in reversed(range(num_resolutions)):
            expected_skip_channels = model_channels * channel_mult[i]
            # Upsample layer if not the first upsampling stage (i.e. if we are not at bottleneck resolution)
            if i != num_resolutions -1 : # if its not the first layer of upsampling path (directly after bottleneck)
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
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=1) # Map to output channels (noise)

    def forward(self, x_t, time, context):
        # --- DEBUGGING PRINTS ---
        print(f"DEBUG (UNetConditional.forward): x_t.shape: {x_t.shape}")
        print(f"DEBUG (UNetConditional.forward): context.shape: {context.shape}")
        print(f"DEBUG (UNetConditional.forward): x_t.shape[2:]: {x_t.shape[2:]}")
        # --- END DEBUGGING PRINTS ---
        """
        Args:
            x_t (torch.Tensor): Noisy target image (B, in_target_channels, H, W)
            time (torch.Tensor): Timesteps (B,)
            context (torch.Tensor): Conditioning data (B, in_condition_channels, H, W)
        Returns:
            torch.Tensor: Predicted noise (B, out_channels, H, W)
        """
        #print("0. model input x_t:", torch.isnan(x_t).any()) #gpt debug
        #print("0. model input context:", torch.isnan(context).any()) #gpt debug

        # 1. Concatenate noisy target and conditioning data
        # Ensure context is resized if necessary (though typically it should match x_t's H, W)
        if x_t.shape[2:] != context.shape[2:]:
             # This is a basic resize, consider more sophisticated alignment if aspects differ
            context = F.interpolate(context, size=x_t.shape[2:], mode='bilinear', align_corners=False)

        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0) # <--- ADD THIS
        nn_input = torch.cat((x_t, context), dim=1)

        #print("1. nn_input:", torch.isnan(nn_input).any()) #gpt debug

        # 2. Compute time embedding
        t_emb = self.time_mlp(time) # (B, time_emb_dim)

        #print("2. time embedding:", torch.isnan(t_emb).any()) #gpt debug

        # 3. Initial convolution
        h = self.init_conv(nn_input) # (B, model_channels, H, W)
        
        #print("✅ After init_conv:", "has_nan:", torch.isnan(h).any().item(), "min:", h.min().item(), "max:", h.max().item()) #gpt debug
        #print("3. init conv output:", torch.isnan(h).any()) #gpt debug

        # Skip connections
        skips = [h] # Store the output of init_conv as the first "skip"

        # 4. Downsampling path
        # Iterate through down_blocks: ResNet -> ResNet -> ... -> DownsampleConv
        block_idx = 0
        num_resolutions = len(self.down_blocks) // (self.num_res_blocks +1) # +1 for downsample layer // gpt wanted me to add self to the num_res_blocks
        for i in range(len(self.channel_mult)): # Iterate through resolutions // added self to channel_mult bc gpt told me to 
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, t_emb)
                #print(f"🔽 Down block {block_idx} output has_nan:", torch.isnan(h).any().item()) #gpt debug
                #print(f"4. down block {block_idx} output:", torch.isnan(h).any()) #gpt debug
                skips.append(h)
                block_idx +=1
            if i < len(self.channel_mult) -1 : # If not the last resolution
                h = self.down_blocks[block_idx](h) # Downsample conv
                #print(f"🔽 Downsample block {block_idx} output has_nan:", torch.isnan(h).any().item()) #gpt debug
                #print(f"5. downsample block {block_idx} output:", torch.isnan(h).any()) #gpt debug
                skips.append(h) # Also store downsampled output before next resnet block
                block_idx +=1


        # 5. Bottleneck
        h = self.mid_block1(h, t_emb)
        #print("🧱 Mid block 1 has_nan:", torch.isnan(h).any().item()) #gpt debug
        h = self.mid_block2(h, t_emb)
        #print("🧱 Mid block 2 has_nan:", torch.isnan(h).any().item()) #gpt debug

        # 6. Upsampling path
        # Iterate through up_blocks: UpsampleConv -> ResNet (cat skip) -> ResNet -> ...
        block_idx = 0
        for i in reversed(range(len(self.channel_mult))):
            if i < len(self.channel_mult) -1 : # If not the first upsampling layer (after bottleneck)
                h = self.up_blocks[block_idx](h) # Upsample conv
                #print(f"🔼 Up block {block_idx} output has_nan:", torch.isnan(h).any().item()) #gpt debug
                block_idx +=1

            # Concatenate with skip connection. Skips are stored in reverse order of use
            # Number of ResNet blocks per resolution is num_res_blocks.
            # There's one additional ResNet block that receives the concatenated features.
            """
            for j in range(self.num_res_blocks +1):
                skip_h = skips.pop()
                # print(f"Up block {block_idx}: h shape: {h.shape}, skip_h shape: {skip_h.shape}")
                if h.size(2) != skip_h.size(2) or h.size(3) != skip_h.size(3): # Ensure spatial dims match for concat
                     skip_h = F.interpolate(skip_h, size=h.shape[2:], mode='bilinear', align_corners=False)

                concat_h = torch.cat((h, skip_h), dim=1)
                h = self.up_blocks[block_idx](concat_h, t_emb) # This should be the ResNet block
                block_idx +=1
            """
            #gpt wanted me to replace that block with this:
            for j in range(self.num_res_blocks + 1):
                skip_h = skips.pop()
                if h.size(2) != skip_h.size(2) or h.size(3) != skip_h.size(3): 
                    skip_h = F.interpolate(skip_h, size=h.shape[2:], mode='bilinear', align_corners=False)

                if j == 0:
                    # First block in upsampling stage: concatenate skip connection
                    h = self.up_blocks[block_idx](torch.cat((h, skip_h), dim=1), t_emb)
                    #print(f"🔼 Up block {block_idx} output has_nan:", torch.isnan(h).any().item()) #gpt debug
                else:
                    # Subsequent blocks already assume the correct number of input channels
                    h = self.up_blocks[block_idx](h, t_emb)
                    #print(f"🔼 Up block {block_idx} output has_nan:", torch.isnan(h).any().item()) #gpt debug
                block_idx += 1


        # 7. Final layer - according to gpt debug
        output = self.final_conv(h)
        """
        print("🟧 UNet final output stats:",
            "min:", output.min().item(),
            "max:", output.max().item(),
            "mean:", output.mean().item(),
            "std:", output.std().item(),
            "has_nan:", torch.isnan(output).any().item())
        """
        return output

        #return self.final_conv(h)


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
    def __init__(self, model, image_size, timesteps=20, beta_schedule_type='linear',
                 target_channels=1, device='cpu'):
        super().__init__()
        self.model = model # The UNet model
        self.image_size = image_size
        self.target_channels = target_channels # Channels of the image being diffused (e.g., 1 for mask)
        self.timesteps = timesteps
        self.device = device

        if beta_schedule_type == 'linear':
            betas = linear_beta_schedule(timesteps).to(self.device)
        elif beta_schedule_type == 'cosine':
            betas = cosine_beta_schedule(timesteps).to(self.device)
        else:
            raise ValueError(f"Unknown beta schedule type: {beta_schedule_type}")
        
        self.betas = betas  # Store betas for use in p_sample // gpt told me to add this


        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_cumprod_t-1
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clip variance to avoid 0, which can happen at t=0 for some schedules
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """Extracts values from a at specified timesteps t and reshapes for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t) # Gathers along the last dimension
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Samples x_t by adding noise to x_0.
        x_start: Original clean image (B, C, H, W)
        t: Timestep (B,)
        noise: Optional noise tensor; if None, generated from N(0,1)
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

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
            noise = torch.randn_like(x_start, device=self.device)

        #print("x_start has nan:", torch.isnan(x_start).any()) #gpt debug
        #print("t has nan:", torch.isnan(t).any()) #gpt debug
        #print("noise has nan:", torch.isnan(noise).any()) #gpt debug

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        #print("x_noisy has nan:", torch.isnan(x_noisy).any()) #gpt debug

        #if torch.isnan(x_noisy).any(): #gpt debug
            #print("NaN detected in x_noisy") #gpt debug

        predicted_noise = self.model(x_noisy, t, context) # UNet predicts the noise

        #if torch.isnan(predicted_noise).any(): #gpt debug
            #print("NaN detected in predicted_noise") #gpt debug

        #if torch.isnan(noise).any(): #gpt debug
            #print("NaN detected in noise") #gpt debug

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        #if torch.isnan(loss).any(): #gpt debug
            #print("NaN detected in loss") #gpt debug

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, context, t_index):
        """
        Reverse diffusion process: p_theta(x_{t-1} | x_t)
        Samples x_{t-1} from x_t using the learned model to predict noise.
        This version uses a more stable formulation by first predicting x_0 and clipping it.
        """
        # Get the numbers needed for the calculation
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # 1. Have the model predict the noise from the current fuzzy image
        predicted_noise = self.model(x_t, t, context)
        
        # 2. Use the noise to guess what the final, clean image (x_0) looks like
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        # 3. THIS IS THE FIX: Apply the "speed limiter".
        #    Clip the guess to ensure its values stay in a safe [-1, 1] range.
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # 4. Now, use this safe, clipped guess to calculate the next, slightly-less-fuzzy image.
        #    This calculation is now stable and won't produce infinite values.
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        model_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t
    
        if t_index == 0:
            # At the last step, we're done. No more noise.
            return model_mean
        else:
            # For all other steps, add the correct, small amount of noise for the next step.
            posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.exp(0.5 * posterior_log_variance_t) * noise
            
    @torch.no_grad()
    def sample(self, context, batch_size=1, channels=1):
        """
        Full sampling loop (Algorithm 2 from DDPM paper).
        Generates new images from noise, conditioned on context.
        context: Conditioning data (B, C_cond, H, W)
        """
        image_shape = (batch_size, channels, self.image_size, self.image_size)
        img = torch.randn(image_shape, device=self.device) # Start with pure noise x_T
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            t_tensor = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t_tensor, context, i)
            #if i % (self.timesteps // 10) == 0 or i < 10: # Save some intermediate steps
            imgs.append(img.cpu())
        return img.cpu(), imgs # Return final image and intermediate steps
