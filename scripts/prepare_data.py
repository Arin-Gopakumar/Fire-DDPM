# In scripts/prepare_data.py

from __future__ import annotations
import argparse
import json
import re
import os
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from typing import Dict, List, Tuple
import hashlib
from tqdm import tqdm

# Regular expression to find a date pattern like "2019-04-19"
date = re.compile(r"(\d{4}-\d{2}-\d{2})")

def resize(img, shape, nearest=False):
    """Resizes an image (numpy array) to a new shape."""
    res = np.empty((img.shape[0], *shape), dtype=img.dtype)
    for i in range(img.shape[0]):
        im = Image.fromarray(img[i])
        method = Image.NEAREST if nearest else Image.BILINEAR
        res[i] = np.asarray(im.resize(shape[::-1], method))
    return res

def read_tif(p):
    """Reads a .tif file and returns its content as a numpy array."""
    with rasterio.open(p) as f:
        return f.read()

def gather_fire_dirs(root: Path) -> Dict[str, List[Path]]:
    """
    Gathers all fire directories and splits them into train, val, and test sets
    based on a hash of the fire's directory name.
    """
    fires = {"train": [], "val": [], "test": []}
    all_fire_events = [fd for year_dir in root.glob("[12][0-9][0-9][0-9]") if year_dir.is_dir() for fd in year_dir.iterdir() if fd.is_dir()]
    
    for fd in sorted(all_fire_events):
        hash_val = int(hashlib.md5(fd.name.encode()).hexdigest(), 16)
        split = "train" if hash_val % 10 < 8 else ("val" if hash_val % 10 < 9 else "test")
        fires[split].append(fd)
        
    print(f"Data split: {len(fires['train'])} train, {len(fires['val'])} val, {len(fires['test'])} test fire events.")
    return fires

def sort_date_tifs(fire_dir):
    """Sorts .tif files in a directory by date."""
    return sorted([p for p in fire_dir.glob("*.tif") if date.search(p.stem)], key=lambda p: date.search(p.stem).group(1))

def collect_samples(fire_dir, k):
    """Creates input/target samples of k days from a single fire event."""
    tifs = sort_date_tifs(fire_dir)
    if len(tifs) < k + 1: return []
    return [(f"{fire_dir.name}_{date.search(past[-1].stem).group(1)}", past, tifs[idx + 1]) for idx, past in enumerate((tifs[i-k+1:i+1] for i in range(k-1, len(tifs)-1)), start=k-1)]

# In scripts/prepare_data.py, replace the existing calculate_global_stats function with this one.

def calculate_global_stats(fires: List[Path], k: int, num_channels_per_day: int) -> Dict:
    """
    Calculates the min and max for each channel across the entire training dataset.
    This version handles cases where a channel may have no valid data.
    """
    print("Calculating global normalization statistics from the training set...")
    # Get a sample to determine the number of channels
    first_sample_path = sort_date_tifs(fires[0])[0]
    num_channels_per_day = read_tif(first_sample_path).shape[0]
    total_channels = k * num_channels_per_day
    print(f"Detected {num_channels_per_day} channels per day, for a total of {total_channels} input channels.")

    channel_mins = [np.inf] * total_channels
    channel_maxs = [-np.inf] * total_channels

    for fd in tqdm(fires, desc="Calculating Stats"):
        for _, imgs, _ in collect_samples(fd, k):
            try:
                stacked_data = np.concatenate([read_tif(p) for p in imgs], 0)
                if stacked_data.shape[0] != total_channels: continue

                for i in range(total_channels):
                    channel_mins[i] = min(channel_mins[i], np.min(stacked_data[i]))
                    channel_maxs[i] = max(channel_maxs[i], np.max(stacked_data[i]))
            except Exception as e:
                # This can happen if a .tif file is corrupt or has unexpected dimensions
                # print(f"Skipping a file during stat calculation due to error: {e}")
                pass

    # --- THIS IS THE FIX ---
    # After calculating, check for any remaining infinity values and replace them.
    for i in range(total_channels):
        if channel_mins[i] == np.inf or channel_maxs[i] == -np.inf:
            print(f"Warning: No valid data found for channel {i}. Using 0.0 as the default min/max.")
            channel_mins[i] = 0.0
            channel_maxs[i] = 0.0
    # --- END OF FIX ---

    return {"mins": channel_mins, "maxs": channel_maxs}

def normalize_globally(x, stats):
    """Normalizes each channel of a numpy array to [-1, 1] using pre-computed global stats."""
    x = x.astype(np.float32)
    mins, maxs = stats["mins"], stats["maxs"]
    for i in range(x.shape[0]):
        the_min, the_max = mins[i], maxs[i]
        if the_max > the_min:
            x[i] = 2 * (x[i] - the_min) / (the_max - the_min) - 1
        else:
            x[i] = 0
    return x

def run(split: str, fires: List[Path], out: Path, k: int, sz: Tuple[int,int], stats: Dict):
    """Processes and saves the data for one split using global normalization stats."""
    if not fires: return
    inp_dir, tgt_dir = out/split/"inputs", out/split/"targets"
    inp_dir.mkdir(parents=True, exist_ok=True); tgt_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    for fd in tqdm(fires, desc=f"Processing {split} split"):
        for sid, imgs, mask_src in collect_samples(fd, k):
            try:
                arrs = [resize(read_tif(p), sz, nearest=False) for p in imgs]
                x = np.concatenate(arrs, 0)
                
                # Apply global normalization
                x = normalize_globally(x, stats)
                np.save(inp_dir/f"{sid}.npy", x)

                mask = read_tif(mask_src)[0]
                mask = resize(mask[None], sz, nearest=True)[0]
                mask = (mask > 0).astype(np.uint8) * 255
                Image.fromarray(mask).save(tgt_dir / f"{sid}.png")
                total += 1
            except Exception as e:
                print(f"Skipping sample {sid} due to error: {e}")
    print(f"Processed {total} samples for the {split} split.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True, help="Path to the raw WildfireSpreadTS dataset directory.")
    p.add_argument("--out", type=Path, default=Path("./data"), help="Path to the output directory for processed data.")
    p.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
    p.add_argument("--days", type=int, default=3, help="Number of past days to use as conditioning.")
    args = p.parse_args()
    
    # New: Path for storing normalization stats
    stats_path = args.out / "normalization_stats.json"

    fires_by_split = gather_fire_dirs(args.raw)
    
    # New logic: Calculate stats only if they don't exist
    if not stats_path.exists():
        # A sample tif is needed to find the number of channels per day
        sample_tif_path = sort_date_tifs(fires_by_split['train'][0])[0]
        num_channels_per_day = read_tif(sample_tif_path).shape[0]
        
        stats = calculate_global_stats(fires_by_split['train'], args.days, num_channels_per_day)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved global normalization stats to {stats_path}")
    else:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded global normalization stats from {stats_path}")

    for sp in ["train", "val", "test"]:
        run(sp, fires_by_split[sp], args.out, args.days, tuple(args.image_size), stats)

if __name__ == "__main__":
    main()
