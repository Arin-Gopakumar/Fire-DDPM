import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import logging
from tqdm import tqdm
import math

# Adjust import paths based on your project structure
# Assuming train.py is in wildfire_ddpm/scripts/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ddpm_model import UNetConditional, GaussianDiffusion
from utils.dataset_loader import WildfireDataset

# --- Configuration ---
# These can be overridden by command-line arguments
CONFIG = {
    "data_dir": "../Fire-DDPM/data_2",
    "checkpoints_dir": "../checkpoints",
    "results_dir": "../outputs/training_samples", # For saving sample images during training
    "run_name": "ddpm_wildfire_run1",
    "epochs": 100, # Adjust as needed
    "batch_size": 4, # Adjust based on GPU memory
    "learning_rate": 1e-4, # Common starting point for Adam
    "image_size": 64, # Must match data preparation and UNet config
    "target_channels": 1, # Wildfire mask is single channel
    "condition_channels": 24, # Example: 3 days * 8 env variables/day. MUST MATCH YOUR DATA.
                                # This is TOTAL_INPUT_CHANNELS from prepare_data.py
    "model_channels": 64, # Base channels for UNet
    "num_res_blocks": 2,
    "channel_mult": (1, 2, 4, 8), # For UNet depth
    "time_emb_dim_mult": 4,
    "unet_groups": 8, # GroupNorm groups
    "diffusion_timesteps": 1000,
    "beta_schedule": "linear", # 'linear' or 'cosine'
    "save_every_n_epochs": 10, # How often to save checkpoints
    "sample_every_n_epochs": 5, # How often to generate and save sample images
    "num_samples_to_generate": 4, # Number of images to generate for visual check
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2, # For DataLoader
    "load_checkpoint_path": None, # Path to a .pt file to resume training
}

def setup_logging(config):
    log_dir = os.path.join(config["checkpoints_dir"], config["run_name"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging setup complete. Log file: {os.path.join(log_dir, 'training.log')}")
    logging.info(f"Configuration: {config}")

def train(config):
    setup_logging(config)
    device = torch.device(config["device"])

    # 1. Dataset and DataLoader
    logging.info("Loading dataset...")
    # Note: Input normalization should primarily be handled in prepare_data.py
    # The WildfireDataset uses basic ToTensor transforms by default.
    train_dataset = WildfireDataset(
        data_dir=config["data_dir"],
        split="train",
        image_size=(config["image_size"], config["image_size"])
    )
    # You might want a validation dataset too for monitoring
    # val_dataset = WildfireDataset(data_dir=config["data_dir"], split="val", image_size=(config["image_size"], config["image_size"]))
    
    if len(train_dataset) == 0:
        logging.error("Training dataset is empty. Please check data preparation and paths.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True # Important for some distributed training, good practice
    )
    logging.info(f"Training dataset loaded: {len(train_dataset)} samples.")

    # 2. Model Instantiation
    logging.info("Initializing models...")
    unet_model = UNetConditional(
        image_size=config["image_size"],
        in_target_channels=config["target_channels"],
        in_condition_channels=config["condition_channels"], # Make sure this matches your data
        model_channels=config["model_channels"],
        out_channels=config["target_channels"], # Predicts noise for the target
        num_res_blocks=config["num_res_blocks"],
        channel_mult=config["channel_mult"],
        time_emb_dim_mult=config["time_emb_dim_mult"],
        groups=config["unet_groups"]
    ).to(device)

    diffusion_process = GaussianDiffusion(
        model=unet_model,
        image_size=config["image_size"],
        timesteps=config["diffusion_timesteps"],
        beta_schedule_type=config["beta_schedule"],
        target_channels=config["target_channels"],
        device=device
    )
    logging.info(f"UNet model parameters: {sum(p.numel() for p in unet_model.parameters() if p.requires_grad)}")

    # 3. Optimizer
    optimizer = optim.AdamW(unet_model.parameters(), lr=config["learning_rate"])
    
    start_epoch = 0
    if config["load_checkpoint_path"] and os.path.exists(config["load_checkpoint_path"]):
        logging.info(f"Loading checkpoint from {config['load_checkpoint_path']}")
        checkpoint = torch.load(config["load_checkpoint_path"], map_location=device)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")


    # 4. Training Loop
    logging.info(f"Starting training on {device} for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        unet_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            targets = batch["target"].to(device) # (B, target_channels, H, W), should be [0,1]
            conditions = batch["condition"].to(device) # (B, condition_channels, H, W)
            
            # DDPMs often work best with inputs normalized to [-1, 1]
            # Targets (masks) are [0, 1]. We can shift them to [-1, 1] for diffusion.
            targets_scaled = (targets * 2) - 1 

            # Generate random timesteps for each sample in the batch
            t = torch.randint(0, config["diffusion_timesteps"], (targets.shape[0],), device=device).long()

            loss = diffusion_process.p_losses(x_start=targets_scaled, t=t, context=conditions, loss_type="l2")
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{config['epochs']} - Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["save_every_n_epochs"] == 0 or (epoch + 1) == config["epochs"]:
            checkpoint_path = os.path.join(config["checkpoints_dir"], config["run_name"], f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': config
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Generate and save sample images
        if (epoch + 1) % config["sample_every_n_epochs"] == 0 or (epoch + 1) == config["epochs"]:
            unet_model.eval()
            with torch.no_grad():
                # Use a fixed batch of conditions for consistent sampling visualization (e.g., from val set)
                # For simplicity, take first N conditions from current training batch if available, or make dummy
                sample_conditions = conditions[:config["num_samples_to_generate"]]
                if sample_conditions.shape[0] == 0 and len(train_dataset) > 0: # if last batch was too small
                    sample_batch = next(iter(train_loader))
                    sample_conditions = sample_batch["condition"][:config["num_samples_to_generate"]].to(device)
                
                if sample_conditions.shape[0] > 0:
                    logging.info(f"Generating {sample_conditions.shape[0]} samples...")
                    # Sample returns final image scaled [-1, 1] and list of intermediates
                    generated_samples_scaled, _ = diffusion_process.sample(
                        context=sample_conditions,
                        batch_size=sample_conditions.shape[0],
                        channels=config["target_channels"]
                    )
                    # Scale back to [0, 1] for saving as image
                    generated_samples_01 = (generated_samples_scaled + 1) / 2.0
                    generated_samples_01 = torch.clamp(generated_samples_01, 0.0, 1.0)

                    save_image(
                        generated_samples_01,
                        os.path.join(config["results_dir"], f"sample_epoch_{epoch+1}.png"),
                        nrow=int(math.sqrt(sample_conditions.shape[0])) # Arrange in a grid
                    )
                    logging.info(f"Saved sample generations to {config['results_dir']}")
                else:
                    logging.warning("Could not generate samples: no conditioning data available for sampling.")
            unet_model.train()

    logging.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional DDPM for Wildfire Spread")
    parser.add_argument("--run_name", type=str, help="Name for the training run")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--image_size", type=int, help="Image size (H and W)")
    parser.add_argument("--condition_channels", type=int, help="Number of channels in conditioning input")
    parser.add_argument("--timesteps", type=int, help="Number of diffusion timesteps")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to resume training")

    args = parser.parse_args()

    # Update config with command line arguments if provided
    if args.run_name: CONFIG["run_name"] = args.run_name
    if args.epochs: CONFIG["epochs"] = args.epochs
    if args.batch_size: CONFIG["batch_size"] = args.batch_size
    if args.lr: CONFIG["learning_rate"] = args.lr
    if args.image_size: CONFIG["image_size"] = args.image_size
    if args.condition_channels: CONFIG["condition_channels"] = args.condition_channels
    if args.timesteps: CONFIG["diffusion_timesteps"] = args.timesteps
    if args.load_checkpoint: CONFIG["load_checkpoint_path"] = args.load_checkpoint
    
    # Ensure checkpoint directory for the run exists
    os.makedirs(os.path.join(CONFIG["checkpoints_dir"], CONFIG["run_name"]), exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    train(CONFIG)