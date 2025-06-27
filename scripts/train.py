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
    train_dataset = WildfireDataset(
        data_dir=config["data_dir"],
        split="train",
        image_size=(config["image_size"], config["image_size"])
    )
    if len(train_dataset) == 0:
        logging.error("Training dataset is empty. Please check data preparation and paths.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"Training dataset loaded: {len(train_dataset)} samples.")

    # <--- STEP 1: LOAD VALIDATION DATASET --->
    logging.info("Loading validation dataset...")
    val_dataset = WildfireDataset(
        data_dir=config["data_dir"],
        split="val",
        image_size=(config["image_size"], config["image_size"])
    )
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"] * 2, # Often can use a larger batch size for validation
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True
        )
        logging.info(f"Validation dataset loaded: {len(val_dataset)} samples.")
    else:
        logging.warning("Validation dataset is empty. Proceeding without validation loop.")
        val_loader = None
    # <--- END OF STEP 1 --->

    # 2. Model Instantiation
    logging.info("Initializing models...")
    unet_model = UNetConditional(
        image_size=config["image_size"],
        in_target_channels=config["target_channels"],
        in_condition_channels=config["condition_channels"],
        model_channels=config["model_channels"],
        out_channels=config["target_channels"],
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

    # <--- STEP 2: INITIALIZE VARIABLE TO TRACK BEST VALIDATION LOSS --->
    best_val_loss = float('inf')

    # 4. Training Loop
    logging.info(f"Starting training on {device} for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        # --- Training Phase ---
        unet_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()
            targets = batch["target"].to(device)
            conditions = batch["condition"].to(device)
            targets_scaled = (targets * 2) - 1 
            t = torch.randint(0, config["diffusion_timesteps"], (targets.shape[0],), device=device).long()
            loss = diffusion_process.p_losses(x_start=targets_scaled, t=t, context=conditions, loss_type="l2")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # <--- STEP 3: ADD VALIDATION LOOP --->
        avg_val_loss = None
        if val_loader:
            unet_model.eval() # Set model to evaluation mode
            epoch_val_loss = 0.0
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False)
            with torch.no_grad(): # Disable gradients for validation
                for batch in val_progress_bar:
                    targets = batch["target"].to(device)
                    conditions = batch["condition"].to(device)
                    targets_scaled = (targets * 2) - 1
                    t = torch.randint(0, config["diffusion_timesteps"], (targets.shape[0],), device=device).long()
                    loss = diffusion_process.p_losses(x_start=targets_scaled, t=t, context=conditions, loss_type="l2")
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)
            logging.info(f"Epoch {epoch+1}/{config['epochs']} -> Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            logging.info(f"Epoch {epoch+1}/{config['epochs']} -> Train Loss: {avg_epoch_loss:.4f}")
        # <--- END OF STEP 3 --->

        # <--- STEP 4: MODIFY CHECKPOINT SAVING LOGIC --->
        # Save a checkpoint only if the validation loss has improved.
        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(config["checkpoints_dir"], config["run_name"], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            logging.info(f"✅ New best model saved to {checkpoint_path} with Val Loss: {best_val_loss:.4f}")
        # <--- END OF STEP 4 --->

        # (Optional) Keep periodic saving of samples for visual inspection
        if (epoch + 1) % config["sample_every_n_epochs"] == 0 or (epoch + 1) == config["epochs"]:
            unet_model.eval()
            with torch.no_grad():
                sample_conditions = conditions[:config["num_samples_to_generate"]]
                if sample_conditions.shape[0] > 0:
                    generated_samples_scaled, _ = diffusion_process.sample(
                        context=sample_conditions,
                        batch_size=sample_conditions.shape[0],
                        channels=config["target_channels"]
                    )
                    generated_samples_01 = (generated_samples_scaled + 1) / 2.0
                    save_image(
                        generated_samples_01,
                        os.path.join(config["results_dir"], f"sample_epoch_{epoch+1}.png"),
                    )
                    logging.info(f"Saved sample generations to {config['results_dir']}")
            # The model is set back to train mode at the start of the next epoch loop.

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
