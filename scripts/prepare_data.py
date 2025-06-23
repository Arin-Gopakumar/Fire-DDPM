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

def normalize(x):
    """Normalizes each channel of a numpy array to the [-1, 1] range."""
    x = x.astype(np.float32)
    for i in range(x.shape[0]):
        the_min = x[i].min()
        the_max = x[i].max()
        if the_max != the_min:
            x[i] = 2 * (x[i] - the_min) / (the_max - the_min) - 1
        else:
            x[i] = 0
    return x

def read_tif(p):
    """Reads a .tif file and returns its content as a numpy array."""
    with rasterio.open(p) as f:
        return f.read()

def gather_fire_dirs(root: Path) -> Dict[str, List[Path]]:
    """
    Gathers all fire directories and splits them into train, val, and test sets
    based on a hash of the fire's directory name. This ensures all data from
    a single fire event remains in the same split.
    """
    fires = {"train": [], "val": [], "test": []}
    all_fire_events = []

    # First, collect all unique fire event directories from all year subfolders
    for year_dir in root.glob("[12][0-9][0-9][0-9]"):
        if not year_dir.is_dir():
            continue
        for fd in year_dir.iterdir():
            if fd.is_dir():
                all_fire_events.append(fd)
    
    # Now, split them based on a hash of their name for reproducibility
    for fd in sorted(all_fire_events):
        # Use a simple hash to deterministically assign each fire to a split
        # This gives an approximate 80/10/10 split
        hash_val = int(hashlib.md5(fd.name.encode()).hexdigest(), 16)
        if hash_val % 10 < 8: # 80% for training
            split = "train"
        elif hash_val % 10 < 9: # 10% for validation
            split = "val"
        else: # 10% for testing
            split = "test"
        fires[split].append(fd)
        
    print(f"Data split: {len(fires['train'])} train, {len(fires['val'])} val, {len(fires['test'])} test fire events.")
    return fires

def sort_date_tifs(fire_dir):
    """Sorts .tif files in a directory by date."""
    return sorted([p for p in fire_dir.glob("*.tif") if date.search(p.stem)],
                  key=lambda p: date.search(p.stem).group(1))

def collect_samples(fire_dir, k):
    """Creates input/target samples of k days from a single fire event."""
    tifs = sort_date_tifs(fire_dir)
    if len(tifs) < k + 1:
        return []
    samples = []
    for idx in range(k - 1, len(tifs) - 1):
        past = tifs[idx - k + 1: idx + 1]
        nxt = tifs[idx + 1]
        sid = f"{fire_dir.name}_{date.search(past[-1].stem).group(1)}"
        samples.append((sid, past, nxt))
    return samples

def run(split: str, fires: List[Path], out: Path, k: int, sz: Tuple[int,int], norm: bool):
    """Processes and saves the data for one split (train, val, or test)."""
    if not fires:
        return
    inp_dir, tgt_dir = out/split/"inputs", out/split/"targets"
    inp_dir.mkdir(parents=True, exist_ok=True); tgt_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    for fd in fires:
        for sid, imgs, mask_src in collect_samples(fd, k):
            arrs = [resize(read_tif(p), sz, nearest=False) for p in imgs]
            x = np.concatenate(arrs, 0)
            if norm:
                x = normalize(x)
            
            # Save the input file.
            np.save(inp_dir/f"{sid}.npy", x)

            # Read the mask. We NO LONGER skip empty masks.
            mask = read_tif(mask_src)[0]

            # Resize and binarize the mask.
            mask = resize(mask[None], sz, nearest=True)[0]
            mask = (mask > 0).astype(np.uint8) * 255

            # Save the processed mask.
            Image.fromarray(mask).save(tgt_dir / f"{sid}.png")
            total += 1
    print(f"Processed {total} samples for the {split} split.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True, help="Path to the raw WildfireSpreadTS dataset directory.")
    p.add_argument("--out", type=Path, default=Path("./data"), help="Path to the output directory for processed data.")
    p.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
    p.add_argument("--days", type=int, default=3, help="Number of past days to use as conditioning.")
    p.add_argument("--no-normalise", action="store_true", help="Disable normalization of input data to [-1, 1].")
    args = p.parse_args()

    # Use the new robust data splitting function.
    fires_by_split = gather_fire_dirs(args.raw)
    
    for sp in ["train", "val", "test"]:
        run(sp, fires_by_split[sp], args.out, args.days, tuple(args.image_size), not args.no_normalise)

if __name__ == "__main__":
    main()
