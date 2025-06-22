# In Fire-DDPM/scripts/evaluate.py

import os
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score

# This import structure works with your project layout
import sys
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

# In scripts/evaluate.py
def calculate_metrics(pred_probs, pred_binary, target_binary):
    """
    Calculates segmentation metrics for a single sample. This function is vital for your paper's results.

    Args:
        pred_probs (torch.Tensor): The model's raw output probabilities, shape (H, W), values in [0, 1].
        pred_binary (torch.Tensor): The binarized prediction mask {0, 1}.
        target_binary (torch.Tensor): The ground truth mask {0, 1}.
    """
    pred_flat_binary = pred_binary.flatten().cpu().numpy()
    target_flat_binary = target_binary.flatten().cpu().numpy()
    pred_flat_probs = pred_probs.flatten().cpu().numpy()

    # THIS IS THE KEY FIX: The special 'if' block for true negatives has been removed.
    # The standard calculations below will now handle all cases.
    # The precision_score and recall_score with zero_division=0 are robust to this case.

    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection

    # The small epsilon (1e-6) correctly handles the case where the union is 0 (a true negative), resulting in an IoU of 1.0.
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred_binary.sum().item() + target_binary.sum().item() + 1e-6)

    # These functions will correctly return 0.0 for precision and recall in a true negative case.
    precision = precision_score(target_flat_binary, pred_flat_binary, zero_division=0)
    recall = recall_score(target_flat_binary, pred_flat_binary, zero_division=0)

    # This correctly handles the average precision score when the target is empty.
    ap = average_precision_score(target_flat_binary, pred_flat_probs) if np.sum(target_flat_binary) > 0 else 0.0

    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall, 'ap': ap}

def evaluate(config):
    """Main evaluation function, designed to work with your project structure."""
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
        timesteps=model_config.get("diffusion_timesteps", 1000),
        device=device
    )

    logging.info(f"Loading test data from: {config['data_dir']}")
    test_dataset = WildfireDataset(data_dir=config["data_dir"], split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    if len(test_loader) == 0:
        logging.error("Test dataset is empty. Check the path and ensure prepare_data.py has run correctly.")
        return
    logging.info(f"Found {len(test_dataset)} samples in the test set.")

    all_metrics = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 'ap': []}
    progress_bar = tqdm(test_loader, desc="Evaluating on Test Set")

    for batch in progress_bar:
        conditions = batch["condition"].to(device)
        targets = batch["target"].to(device)

        with torch.no_grad():
            generated_samples_scaled, _ = diffusion_process.sample(context=conditions, batch_size=conditions.shape[0])

        if torch.isnan(generated_samples_scaled).any():
            logging.warning(f"NaN value detected in model output for a sample in this batch. Skipping batch.")
            continue

        pred_probs = (generated_samples_scaled.cpu() + 1) / 2.0
        pred_binary = (pred_probs > 0.5).float()

        for i in range(len(conditions)):
            target = targets[i].to(pred_binary.device)
            metrics = calculate_metrics(pred_probs[i], pred_binary[i], target)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

    logging.info("\n" + "="*40)
    logging.info("      FINAL EVALUATION REPORT      ")
    logging.info("="*40)
    for key, scores in all_metrics.items():
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logging.info(f"{key.upper():<18}: {mean_score:.4f} ± {std_score:.4f}")
        else:
            logging.info(f"{key.upper():<18}: No valid samples to score.")
    logging.info("="*40)

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
