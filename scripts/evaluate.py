import os
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import sys
import torchmetrics
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

    unet_model = UNetConditional(
        image_size=model_config.get("image_size", 64),
        in_target_channels=model_config.get("target_channels", 1),
        in_condition_channels=model_config.get("condition_channels", 24),
        model_channels=model_config.get("model_channels", 64),
        out_channels=model_config.get("target_channels", 1),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        channel_mult=tuple(model_config.get("channel_mult", (1, 2, 4, 8))),
    ).to(device)
    unet_model.load_state_dict(checkpoint['model_state_dict'])
    unet_model.eval()

    diffusion_process = GaussianDiffusion(
        model=unet_model,
        image_size=model_config.get("image_size", 64),
        timesteps=model_config.get("diffusion_timesteps", 20),
        device=device
    )

    logging.info(f"Loading test data from: {config['data_dir']}")
    test_dataset = WildfireDataset(data_dir=config["data_dir"], split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    if len(test_loader) == 0:
        logging.error("Test dataset is empty. Check the path and ensure prepare_data.py has run correctly.")
        return
    logging.info(f"Found {len(test_dataset)} samples in the test set.")

    # --- NEW: Initialize torchmetrics objects ---
    # Metrics for all samples
    metrics_all = torchmetrics.MetricCollection({
        'AP': torchmetrics.classification.BinaryAveragePrecision(),
        'Precision': torchmetrics.classification.BinaryPrecision(),
        'Recall': torchmetrics.classification.BinaryRecall(),
        'IoU': torchmetrics.classification.BinaryJaccardIndex(),
        'Dice': torchmetrics.Dice()
    }).to(device)

    # Metrics for samples that contain fire
    metrics_positive = torchmetrics.MetricCollection({
        'AP': torchmetrics.classification.BinaryAveragePrecision(),
        'Precision': torchmetrics.classification.BinaryPrecision(),
        'Recall': torchmetrics.classification.BinaryRecall(),
        'IoU': torchmetrics.classification.BinaryJaccardIndex(),
        'Dice': torchmetrics.Dice()
    }).to(device)
    
    num_positive_samples = 0
    # --- End of new initialization ---

    progress_bar = tqdm(test_loader, desc="Evaluating on Test Set")

    for batch in progress_bar:
        conditions = batch["condition"].to(device)
        targets = batch["target"].to(device) # Shape: (B, 1, H, W), values are 0 or 1

        with torch.no_grad():
            generated_samples_scaled, _ = diffusion_process.sample(context=conditions, batch_size=conditions.shape[0])

        if torch.isnan(generated_samples_scaled).any():
            logging.warning("NaN value detected in model output for a sample in this batch. Skipping batch.")
            continue
        
        # pred_probs are the heatmaps, used for Average Precision
        pred_probs = (generated_samples_scaled.to(device) + 1) / 2.0 # Shape: (B, 1, H, W), values are [0,1]
        # pred_binary is the binarized output, used for other metrics
        pred_binary = (pred_probs > 0.5).int()
        targets_int = targets.int()

        # --- NEW: Update metrics using torchmetrics ---
        # Flatten spatial dimensions to treat each pixel as a sample
        flat_preds_prob = pred_probs.flatten()
        flat_preds_binary = pred_binary.flatten()
        flat_targets = targets_int.flatten()
        
        # Update metrics for all samples
        metrics_all.update(flat_preds_prob, flat_targets)

        # Update metrics for fire-positive samples only
        for i in range(targets.shape[0]):
            if torch.sum(targets[i]) > 0:
                num_positive_samples += 1
                positive_preds_prob_flat = pred_probs[i].flatten()
                positive_preds_binary_flat = pred_binary[i].flatten()
                positive_targets_flat = targets_int[i].flatten()
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
    parser.add_argument("--output_dir", type=str, default="../../outputs/evaluation_reports", help="Directory to save evaluation log file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation. Adjust based on GPU memory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    eval_config = vars(args)
    evaluate(eval_config)
