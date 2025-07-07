import os
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import sys
import torchmetrics
from PIL import Image # Added for image saving

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ddpm_model import UNetConditional, GaussianDiffusion
from utils.dataset_loader import WildfireDataset
from torch.utils.data import DataLoader
import torch

def setup_evaluation_logging(config):
    """Sets up logging for the evaluation script."""
    os.makedirs(config["output_dir"], exist_ok=True)
    log_filename = os.path.join(config["output_dir"], f"evaluation_report_{config['run_name']}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Evaluation log will be saved to: {log_filename}")
    logging.info(f"Evaluation Configuration: {config}")


def evaluate(config):
    """Main evaluation function, rewritten to use torchmetrics for better accuracy."""
    setup_evaluation_logging(config)
    device = torch.device(config["device"])

    logging.info(f"Checkpoint path value received: {config['checkpoint']}")
    if not config["checkpoint"] or not os.path.exists(config["checkpoint"]):
        logging.error("A valid checkpoint path is required.")
        return
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    model_config = checkpoint.get('config', {})
    logging.info(f"Loaded checkpoint from training run: {model_config.get('run_name', 'N/A')}")

    # Retrieve model configuration parameters, with defaults
    image_size = model_config.get("image_size", 64)
    target_channels = model_config.get("target_channels", 1)
    condition_channels_total = model_config.get("condition_channels", 24) # Total input channels (k * channels_per_day)
    model_channels = model_config.get("model_channels", 64)
    num_res_blocks = model_config.get("num_res_blocks", 2)
    channel_mult = tuple(model_config.get("channel_mult", (1, 2, 4, 8)))
    diffusion_timesteps = model_config.get("diffusion_timesteps", 20)
    
    # IMPORTANT ASSUMPTIONS for visualization:
    # 1. Number of input days (k) is assumed to be 3 (default in prepare_data.py)
    #    This 'days' argument should be passed to evaluate.py
    num_input_days = config.get("days", 3) 
    
    # 2. Number of channels per single day's input block
    #    If total condition_channels_total is 24 and num_input_days is 3, then 8 channels per day.
    NUM_CHANNELS_PER_SINGLE_DAY_INPUT = condition_channels_total // num_input_days
    if condition_channels_total % num_input_days != 0:
        logging.warning(f"Total condition channels ({condition_channels_total}) is not divisible by number of input days ({num_input_days}). "
                        "Channel indexing for visualization might be incorrect. Please check your prepare_data.py and model_config.")

    # 3. Index of the active fire channel within a single day's input block
    #    Common convention is that active fire is the first channel (index 0).
    ACTIVE_FIRE_CHANNEL_INDEX_WITHIN_SINGLE_DAY = 0 
    
    # Calculate the index of the active fire channel for the LAST input day (Day N)
    # This channel is part of the 'conditions' tensor.
    ACTIVE_FIRE_CHANNEL_INDEX_IN_CONDITIONS = (num_input_days - 1) * NUM_CHANNELS_PER_SINGLE_DAY_INPUT + ACTIVE_FIRE_CHANNEL_INDEX_WITHIN_SINGLE_DAY
    
    logging.info(f"Assuming {num_input_days} input days, with {NUM_CHANNELS_PER_SINGLE_DAY_INPUT} channels per day.")
    logging.info(f"Active fire channel for Day N is assumed to be at index {ACTIVE_FIRE_CHANNEL_INDEX_IN_CONDITIONS} in the concatenated conditions tensor.")
    logging.info("Please verify these assumptions based on your prepare_data.py and raw data structure.")


    unet_model = UNetConditional(
        image_size=image_size,
        in_target_channels=target_channels,
        in_condition_channels=condition_channels_total,
        model_channels=model_channels,
        out_channels=target_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
    ).to(device)
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()

    diffusion_process = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=diffusion_timesteps,
        device=device
    )

    logging.info(f"Loading test data from: {config['data_dir']}")
    # Pass image_size to WildfireDataset constructor
    test_dataset = WildfireDataset(data_dir=config["data_dir"], split="test", image_size=(image_size, image_size))
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    if len(test_loader) == 0:
        logging.error("Test dataset is empty. Check the path and ensure prepare_data.py has run correctly.")
        return
    logging.info(f"Found {len(test_dataset)} samples in the test set.")

    # --- Initialize torchmetrics objects ---
    # Metrics for all samples
    metrics_all = torchmetrics.MetricCollection({
        'AP': torchmetrics.AveragePrecision(task="binary"),
        'Precision': torchmetrics.Precision(task="binary"),
        'Recall': torchmetrics.Recall(task="binary"),
        'IoU': torchmetrics.JaccardIndex(task="binary"),
        #'Dice': torchmetrics.Dice(task="binary") 
    }).to(device)

    # Metrics for samples that contain fire
    metrics_positive = torchmetrics.MetricCollection({
        'AP': torchmetrics.AveragePrecision(task="binary"),
        'Precision': torchmetrics.Precision(task="binary"),
        'Recall': torchmetrics.Recall(task="binary"),
        'IoU': torchmetrics.JaccardIndex(task="binary"),
        #'Dice': torchmetrics.Dice(task="binary") 
    }).to(device)
    
    num_positive_samples = 0
    # --- End of new initialization ---

    # --- Setup for saving visualizations ---
    viz_output_dir = os.path.join(config["output_dir"], "visualizations", config['run_name'])
    os.makedirs(viz_output_dir, exist_ok=True)
    logging.info(f"Saving example visualizations to: {viz_output_dir}")
    num_samples_to_save = 5 # Save images for the first 5 samples encountered
    samples_saved_count = 0
    # --- End of visualization setup ---

    progress_bar = tqdm(test_loader, desc="Evaluating on Test Set")

    # --- Main evaluation loop ---
    for batch_idx, batch in enumerate(progress_bar):
        conditions = batch["condition"].to(device)
        # targets now contain 0.0 and 2550.0 values from prepare_data.py
        targets = batch["target"].to(device) 
        sample_ids = batch["id"] # Get sample IDs from dataset_loader

        with torch.no_grad():
            generated_samples_scaled, _ = diffusion_process.sample(context=conditions, batch_size=conditions.shape[0])

        if torch.isnan(generated_samples_scaled).any():
            logging.warning("NaN value detected in model output for a sample in this batch. Skipping batch.")
            continue
        
        # pred_probs are the heatmaps, used for Average Precision
        # Model output is [-1, 1], scaled to [0, 1]
        pred_probs = (generated_samples_scaled.to(device) + 1) / 2.0 

        # --- NEW: Binarize targets for metrics and visualization ---
        # Assuming 0.0 means "fire" (positive class) and 2550.0 means "no fire" (negative class)
        # So, target_binary will be 1 for fire, 0 for no fire.
        targets_binary_for_metrics = (targets == 0.0).int() 
        
        # pred_binary for visualization (still uses 0.5 threshold on model's probabilities)
        pred_binary = (pred_probs > 0.5).int() 
        
        # Use the newly binarized targets for flattening
        flat_targets = targets_binary_for_metrics.flatten()
        # --- END NEW ---

        # --- Save visualizations for a few samples ---
        if samples_saved_count < num_samples_to_save:
            for i in range(conditions.shape[0]): # Iterate through samples in the current batch
                if samples_saved_count >= num_samples_to_save:
                    break # Stop saving if we've reached the limit

                current_sample_id = sample_ids[i]

                # Extract active fire map from the last input day (Day N)
                # Ensure the index is within bounds
                if ACTIVE_FIRE_CHANNEL_INDEX_IN_CONDITIONS < conditions.shape[1]:
                    # Assuming active fire map is a single channel
                    input_fire_day_N = conditions[i, ACTIVE_FIRE_CHANNEL_INDEX_IN_CONDITIONS, :, :].cpu().numpy()
                    # Scale to 0-255 for saving as image
                    input_fire_day_N_img = (input_fire_day_N * 255).astype(np.uint8)
                    Image.fromarray(input_fire_day_N_img).save(os.path.join(viz_output_dir, f"{current_sample_id}_input_fire_dayN.png"))
                else:
                    logging.warning(f"Could not extract input fire channel for {current_sample_id}. Index {ACTIVE_FIRE_CHANNEL_INDEX_IN_CONDITIONS} out of bounds for conditions with {conditions.shape[1]} channels.")


                # Ground truth target for Day N+1 (now correctly scaled for visualization)
                # If 0.0 is fire (positive), map to 255 (white). If 2550.0 is no fire (negative), map to 0 (black).
                # This makes fire appear white, which is a more common visual convention.
                target_day_N_plus_1_viz = (targets_binary_for_metrics[i, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(target_day_N_plus_1_viz).save(os.path.join(viz_output_dir, f"{current_sample_id}_target_dayN+1.png"))

                # Predicted probabilities for Day N+1
                pred_probs_day_N_plus_1 = pred_probs[i, 0, :, :].cpu().numpy() # Pred is 1 channel
                pred_probs_day_N_plus_1_img = (pred_probs_day_N_plus_1 * 255).astype(np.uint8)
                Image.fromarray(pred_probs_day_N_plus_1_img).save(os.path.join(viz_output_dir, f"{current_sample_id}_pred_probs_dayN+1.png"))

                # Binarized prediction for Day N+1
                pred_binary_day_N_plus_1 = pred_binary[i, 0, :, :].cpu().numpy() # Pred is 1 channel
                pred_binary_day_N_plus_1_img = (pred_binary_day_N_plus_1 * 255).astype(np.uint8)
                Image.fromarray(pred_binary_day_N_plus_1_img).save(os.path.join(viz_output_dir, f"{current_sample_id}_pred_binary_dayN+1.png"))
                
                samples_saved_count += 1
        # --- End of visualization saving ---

        # --- Update metrics using torchmetrics ---
        # Flatten spatial dimensions to treat each pixel as a sample
        flat_preds_prob = pred_probs.flatten()
        # Pass the correctly binarized targets to metrics
        # flat_targets is already defined above
        
        # Update metrics for all samples
        metrics_all.update(flat_preds_prob, flat_targets)

        # Update metrics for fire-positive samples only
        # Check for positive samples based on the *correctly binarized* targets
        for i in range(targets_binary_for_metrics.shape[0]):
            if torch.sum(targets_binary_for_metrics[i]) > 0:
                num_positive_samples += 1
                positive_preds_prob_flat = pred_probs[i].flatten()
                # Use targets_binary_for_metrics for positive samples as well
                positive_targets_flat = targets_binary_for_metrics[i].flatten() 
                metrics_positive.update(positive_preds_prob_flat, positive_targets_flat)
        # --- End of new metric update logic ---

    def log_metric_report(title, metrics_collection, num_samples, total_samples):
        logging.info("\n" + "="*50)
        logging.info(f"      {title}      ")
        logging.info("="*50)
        
        # Compute final metrics from the aggregated state
        final_metrics = metrics_collection.compute()
        
        if num_samples > 0:
            logging.info(f"Evaluated on {num_samples} / {total_samples} samples.")
            for key, score in final_metrics.items():
                logging.info(f"{key:<18}: {score.item():.4f}")
        else:
            logging.info(f"No samples found for this category.")
        logging.info("="*50)

    # Log reports
    log_metric_report("FINAL EVALUATION REPORT (FIRE-POSITIVE SAMPLES)", metrics_positive, num_positive_samples, len(test_dataset))
    log_metric_report("OVERALL EVALUATION REPORT (ALL SAMPLES)", metrics_all, len(test_dataset), len(test_dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Conditional DDPM for Wildfire Spread")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the trained model checkpoint (.pt) created by train.py.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to the processed data directory created by prepare_data.py (the one containing train/val/test folders).")
    parser.add_argument("--run_name", type=str, default="evaluation", help="A name for this evaluation run, for the log file.")
    parser.add_argument("--output_dir", type=str, default="../outputs/evaluation_reports", help="Directory to save evaluation log file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation. Adjust based on GPU memory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--days", type=int, default=3, help="Number of past days used as conditioning input (must match prepare_data.py).")
    args = parser.parse_args()

    eval_config = vars(args)
    evaluate(eval_config)
