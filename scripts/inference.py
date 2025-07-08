import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
import logging
import math
import glob

# Adjust import paths
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ddpm_model import UNetConditional, GaussianDiffusion
from utils.dataset_loader import WildfireDataset # For potential input transforms or loading one sample

# --- Configuration for Inference ---
INFERENCE_CONFIG = {
    "checkpoint_path": None, # REQUIRED: Path to the trained model checkpoint (.pt)
    "output_base_dir": "../outputs/inference_results",
    "num_samples": 1, # Number of masks to generate for the given condition
    "image_size": 64, # Must match the trained model's image size
    "target_channels": 1,
    # These UNet/Diffusion params must match the trained model's config:
    # They are usually stored in the checkpoint, but we'll set defaults here.
    "condition_channels": 24, # MUST match your data and trained model
    "model_channels": 64,
    "num_res_blocks": 2,
    "channel_mult": (1, 2, 4, 8),
    "time_emb_dim_mult": 4,
    "unet_groups": 8,
    "diffusion_timesteps": 20,
    "beta_schedule": "linear",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def setup_inference_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "inference.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Inference logging setup complete. Log file: {os.path.join(output_dir, 'inference.log')}")


def load_conditioning_input(path, image_size, device):
    """Loads and preprocesses a single conditioning input .npy file."""
    try:
        condition_data_np = np.load(path) # Expected (C, H, W)
        
        # Basic validation of shape - C can vary, H,W should be checked if possible
        # Here we assume it's already C, H, W
        
        # Convert to tensor
        condition_tensor = torch.from_numpy(condition_data_np.astype(np.float32))

        # Ensure correct dimensions (add batch dim)
        if condition_tensor.ndim == 3: # (C, H, W)
            condition_tensor = condition_tensor.unsqueeze(0) # (1, C, H, W)
        
        # Resize if necessary (though ideally data is preprocessed to the correct size)
        if condition_tensor.shape[2] != image_size or condition_tensor.shape[3] != image_size:
            logging.warning(f"Condition input size {condition_tensor.shape[2:]} differs from model image_size {image_size}. Resizing.")
            condition_tensor = torch.nn.functional.interpolate(
                condition_tensor, 
                size=(image_size, image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Normalization: If your `prepare_data.py` normalized inputs to [-1, 1],
        # and you saved them like that, then no further normalization is needed here.
        # If they are [0, 1], they might need to be shifted if the model expects [-1,1] for condition.
        # The UNet internally concatenates target (scaled to -1,1) and condition.
        # So, if condition is also scaled to -1,1, that's consistent.
        # Example: if your .npy files were saved as [0,1] range:
        # condition_tensor = (condition_tensor * 2) - 1

        return condition_tensor.to(device)
    except Exception as e:
        logging.error(f"Error loading conditioning input from {path}: {e}")
        raise

def run_inference(config):
    # Determine the specific output directory for this input file
    input_basename = os.path.splitext(os.path.basename(config["condition_input_path"]))[0]
    
    current_output_dir = os.path.join(config["output_base_dir"], input_basename)
    
    # Setup logging for this specific run, creating a new log file per input
    setup_inference_logging(current_output_dir)
    logging.info(f"Inference Configuration for {input_basename}:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    device = torch.device(config["device"])

    if not config["checkpoint_path"] or not os.path.exists(config["checkpoint_path"]):
        logging.error("Checkpoint path is required and must exist.")
        return
    if not config["condition_input_path"] or not os.path.exists(config["condition_input_path"]):
        logging.error("Condition input path is required and must exist.")
        return

    # 1. Load Model Configuration from Checkpoint (if available) or use CLI
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model_config = checkpoint.get('config', {}) # Get training config if saved

    # Override with CLI/defaults if not in checkpoint or if user specifies
    image_size = model_config.get("image_size", config["image_size"])
    condition_channels = model_config.get("condition_channels", config["condition_channels"])
    # ... (load other relevant model params)
    # For simplicity, we'll use the INFERENCE_CONFIG defaults if not in checkpoint,
    # but it's best if the checkpoint stores all necessary architecture details.
    
    logging.info(f"Using image size: {image_size}, condition channels: {condition_channels}")

    # 2. Model Instantiation
    logging.info("Initializing model...")
    unet_model = UNetConditional(
        image_size=image_size,
        in_target_channels=config["target_channels"],
        in_condition_channels=condition_channels,
        model_channels=model_config.get("model_channels", config["model_channels"]),
        out_channels=config["target_channels"],
        num_res_blocks=model_config.get("num_res_blocks", config["num_res_blocks"]),
        channel_mult=tuple(model_config.get("channel_mult", config["channel_mult"])),
        time_emb_dim_mult=model_config.get("time_emb_dim_mult", config["time_emb_dim_mult"]),
        groups=model_config.get("unet_groups", config["unet_groups"])
    ).to(device)
    
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()
    logging.info(f"Model loaded from {config['checkpoint_path']}")

    diffusion_process = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=model_config.get("diffusion_timesteps", config["diffusion_timesteps"]),
        beta_schedule_type=model_config.get("beta_schedule", config["beta_schedule"]),
        target_channels=config["target_channels"],
        device=device
    )

    # 3. Load Conditioning Input
    logging.info(f"Loading conditioning input from {config['condition_input_path']}")
    conditioning_tensor = load_conditioning_input(config["condition_input_path"], image_size, device)
    
    # Repeat conditioning tensor if generating multiple samples for the same condition
    if config["num_samples"] > 1:
        conditioning_tensor = conditioning_tensor.repeat(config["num_samples"], 1, 1, 1)

    # 4. Perform Sampling (Reverse Diffusion)
    logging.info(f"Starting sampling for {config['num_samples']} mask(s)...")
    with torch.no_grad():
        # `sample` returns final image scaled [-1, 1] and list of intermediate steps
        generated_samples_scaled_final, intermediate_steps = diffusion_process.sample(
            context=conditioning_tensor,
            batch_size=config["num_samples"],
            channels=config["target_channels"]
        )
    
    # --- Select 20th timestep for final prediction ---
    # The list intermediate_steps is 0-indexed, so 20th timestep is index 19.
    # Ensure it's not out of bounds.
    if len(intermediate_steps) > 19:
        generated_samples_scaled_for_final_pred = intermediate_steps[19]
        logging.info("Using 20th timestep (index 19) output for final predicted mask.")
    else:
        generated_samples_scaled_for_final_pred = generated_samples_scaled_final
        logging.warning("Less than 20 timesteps available. Using final denoised output for predicted mask.")
    # --- END Selection ---

    # Scale the selected output back to [0, 1] for saving as image (probability map)
    generated_samples_01 = (generated_samples_scaled_for_final_pred + 1) / 2.0
    generated_samples_01 = torch.clamp(generated_samples_01, 0.0, 1.0)
    
    # --- Binarize the output for saving and apply inverted color mapping ---
    # Model predicts high probabilities for the positive class (which is 0.0 in your targets).
    # If model_output > 0.5, it predicts the positive class (fire).
    # We want fire (positive class) to be BLACK (0) and no-fire (negative class) to be WHITE (255).
    generated_samples_binary = (generated_samples_01 > 0.5).float()
    # Invert the binary mask: 1.0 (fire) becomes 0 (black), 0.0 (no fire) becomes 1.0 (white)
    generated_samples_inverted = 1.0 - generated_samples_binary 
    # --- END Binarization and Inversion ---

    # 5. Save Output
    os.makedirs(current_output_dir, exist_ok=True)
    output_basename = os.path.splitext(os.path.basename(config["condition_input_path"]))[0]
    for i in range(config["num_samples"]):
        output_filename = os.path.join(current_output_dir, f"predicted_mask_{output_basename}_sample{i:02d}.png")
        
        # Save the inverted binarized output, scaled to 0 or 255 for PNG
        save_image(generated_samples_inverted[i] * 255, output_filename) 
        logging.info(f"Saved generated mask to {output_filename}")

    # --- Save intermediate steps (grayscale and binarized with inverted colors) ---
    if intermediate_steps and len(intermediate_steps) > 0:
        intermediate_dir = os.path.join(current_output_dir, f"intermediates_{output_basename}_sample00")
        os.makedirs(intermediate_dir, exist_ok=True)
        logging.info(f"Saving intermediate steps to {intermediate_dir}...")
        for step_idx, step_img_scaled in enumerate(intermediate_steps):
            step_img_01 = (step_img_scaled[0] + 1) / 2.0 # Taking the first sample from batch, scale to [0,1]
            step_img_01 = torch.clamp(step_img_01, 0.0, 1.0)

            # Save grayscale heatmap (fire=white, nofire=black)
            save_image(step_img_01 * 255, os.path.join(intermediate_dir, f"step_{step_idx:04d}_grayscale.png")) 

            # Save binarized heatmap with inverted colors (fire=black, nofire=white)
            step_img_binary = (step_img_01 > 0.5).float()
            step_img_binary_inverted = 1.0 - step_img_binary # Invert for visualization
            save_image(step_img_binary_inverted * 255, os.path.join(intermediate_dir, f"step_{step_idx:04d}_binary.png")) 
        logging.info("Intermediate steps saved.")
    # --- END Saving ---

    logging.info("Inference finished for this input file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Wildfire Spread Masks using a trained Conditional DDPM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    # Modify condition_input to not be required
    parser.add_argument("--condition_input", type=str, help="Path to a single conditioning input .npy file")
    # Add new argument for directory
    parser.add_argument("--condition_dir", type=str, help="Path to a directory containing multiple .npy conditioning input files for batch processing")
    
    parser.add_argument("--output_dir", type=str, help="Base directory to save generated masks. Subfolders will be created for each input file.")
    parser.add_argument("--num_samples", type=int, help="Number of masks to generate for the input")
    parser.add_argument("--image_size", type=int, help="Image size (override if not in checkpoint config)")
    parser.add_argument("--condition_channels", type=int, help="Condition channels (override if not in checkpoint config)")

    args = parser.parse_args()

    # Apply command line arguments to a temporary config
    # We copy INFERENCE_CONFIG to avoid modifying the global default during batch processing
    base_config = INFERENCE_CONFIG.copy()
    base_config["checkpoint_path"] = args.checkpoint
    
    if args.output_dir: base_config["output_base_dir"] = args.output_dir
    
    if args.num_samples: base_config["num_samples"] = args.num_samples
    if args.image_size: base_config["image_size"] = args.image_size
    if args.condition_channels: base_config["condition_channels"] = args.condition_channels

    if args.condition_dir:
        if not os.path.isdir(args.condition_dir):
            print(f"Error: Directory '{args.condition_dir}' not found.")
            sys.exit(1)
        
        npy_files = sorted(glob.glob(os.path.join(args.condition_dir, "*.npy")))
        if not npy_files:
            print(f"No .npy files found in '{args.condition_dir}'. Exiting.")
            sys.exit(0)

        print(f"Found {len(npy_files)} .npy files in '{args.condition_dir}'. Starting batch inference.")
        for i, npy_file_path in enumerate(npy_files):
            print(f"\nProcessing file {i+1}/{len(npy_files)}: {os.path.basename(npy_file_path)}")
            # Create a per-file config
            current_config = base_config.copy()
            current_config["condition_input_path"] = npy_file_path
            run_inference(current_config)

    elif args.condition_input:
        if not os.path.exists(args.condition_input):
            print(f"Error: Conditioning input file '{args.condition_input}' not found.")
            sys.exit(1)
        
        base_config["condition_input_path"] = args.condition_input
        print(f"Processing single file: {os.path.basename(args.condition_input)}")
        run_inference(base_config)
    else:
        parser.print_help()
        print("\nError: Either --condition_input or --condition_dir must be provided.")
        sys.exit(1)

    print("\nAll inference tasks completed.")
