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
from datetime import datetime, timedelta
import math # Import math for isnan


# Regular expression to find a date pattern like "2019-04-19"
date = re.compile(r"(\d{4}-\d{2}-\d{2})")


def resize(img, shape, nearest=False):
   """Resizes an image (numpy array) to a new shape.
   This function expects img to be (C, H, W) or (H, W).
   """
   if img.ndim == 3: # (C, H, W)
       res = np.empty((img.shape[0], *shape), dtype=img.dtype)
       for i in range(img.shape[0]):
           im = Image.fromarray(img[i])
           method = Image.NEAREST if nearest else Image.BILINEAR
           res[i] = np.asarray(im.resize(shape[::-1], method))
       return res
   elif img.ndim == 2: # (H, W)
       im = Image.fromarray(img)
       method = Image.NEAREST if nearest else Image.BILINEAR
       return np.asarray(im.resize(shape[::-1], method))
   else:
       raise ValueError("Image must be 2D (H,W) or 3D (C,H,W)")


def read_tif(p):
   """Reads a .tif file and returns its content as a numpy array."""
   with rasterio.open(p) as f:
       return f.read()
      
def gather_fire_dirs(root: Path) -> Dict[str, List[Path]]:
   """
   Return {"train": [...], "val": [...], "test": [...]} where each entry is a list
   of *fire-event folders* (directories that contain all the day-*.tif files
   for that fire).


   This implementation uses a year-based split as requested:
   * Train: All fire events from 2018 and 2019.
   * Test: All fire events from 2020.
   * Val: All fire events from 2021.
   """
   print("Gathering fire directories and splitting by year...")
   fires = {"train": [], "val": [], "test": []}


   # Find all year directories
   year_dirs = [p for p in root.glob("[12][0-9][0-9][0-9]") if p.is_dir()]


   for year_dir in sorted(year_dirs):
       year = year_dir.name
       fire_events_in_year = [fd for fd in year_dir.iterdir() if fd.is_dir()]


       if year in ["2018", "2019"]:
           print(f"Assigning {len(fire_events_in_year)} fire events from {year} to TRAIN set.")
           fires["train"].extend(fire_events_in_year)
       elif year == "2020":
           print(f"Assigning {len(fire_events_in_year)} fire events from {year} to TEST set.")
           fires["test"].extend(fire_events_in_year)
       elif year == "2021"]:
           print(f"Assigning {len(fire_events_in_year)} fire events from {year} to VAL set.")
           fires["val"].extend(fire_events_in_year)
       else:
           print(f"Warning: Year {year} is present in the data but not assigned to a split. Skipping.")


   print(
       f"\nYear-based split complete: "
       f"{len(fires['train'])} train  |  "
       f"{len(fires['val'])} val  |  "
       f"{len(fires['test'])} test fire events."
   )
   return fires


def sort_date_tifs(fire_dir):
   """Sorts .tif files in a directory by date."""
   return sorted([p for p in fire_dir.glob("*.tif") if date.search(p.stem)], key=lambda p: date.search(p.stem).group(1))


def collect_samples(fire_dir, k):
   """
   Creates input/target samples of k days from a single fire event.
   Returns tuples of (input_sample_id, input_tif_paths, target_tif_path, target_sample_id).
  
   input_sample_id: ID based on the last day of the input sequence (Day N).
   input_tif_paths: List of Path objects for the k input days.
   target_tif_path: Path object for the target day (Day N+1).
   target_sample_id: ID based on the target day (Day N+1).
   """
   tifs = sort_date_tifs(fire_dir)
   if len(tifs) < k + 1: return []


   samples = []
   for idx in range(k - 1, len(tifs) - 1):
       past_tifs = tifs[idx - k + 1 : idx + 1] # k input days, ending at tifs[idx] (Day N)
       target_tif = tifs[idx + 1] # The (k+1)-th day, which is Day N+1


       # Input sample ID is based on the date of the LAST input day (Day N)
       input_sid = f"{fire_dir.name}_{date.search(past_tifs[-1].stem).group(1)}"
      
       # Target sample ID is based on the date of the TARGET day (Day N+1)
       target_sid = f"{fire_dir.name}_{date.search(target_tif.stem).group(1)}"
      
       samples.append((input_sid, past_tifs, target_tif, target_sid))
   return samples


def calculate_global_stats(fires: List[Path], k: int, num_channels_per_day: int) -> Dict:
   """
   Calculates the min and max for each channel across the entire training dataset.
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
       for input_sid, imgs, mask_src, target_sid in collect_samples(fd, k):
           try:
               stacked_data = np.concatenate([read_tif(p) for p in imgs], 0)
               # Before calculating stats, replace NaNs with 0.0 so min/max are not NaN
               # This is just for stats calculation, not the final data.
               stacked_data_temp = np.nan_to_num(stacked_data, nan=0.0)
               if stacked_data_temp.shape[0] != total_channels: continue


               for i in range(total_channels):
                   channel_mins[i] = min(channel_mins[i], np.min(stacked_data_temp[i]))
                   channel_maxs[i] = max(channel_maxs[i], np.max(stacked_data_temp[i]))
           except Exception as e:
               pass


   for i in range(total_channels):
       if channel_mins[i] == np.inf or channel_maxs[i] == -np.inf:
           print(f"Warning: No valid data found for channel {i}. Using 0.0 as the default min/max.")
           channel_mins[i] = 0.0
           channel_maxs[i] = 0.0


   channel_mins = [float(v) for v in channel_mins]
   channel_maxs = [float(v) for v in channel_maxs]


   return {"mins": channel_mins, "maxs": channel_maxs}


def normalize_globally(x, stats):
   """Normalizes each channel of a numpy array to [-1, 1] using pre-computed global stats."""
   x = x.astype(np.float32)
   mins, maxs = stats["mins"], stats["maxs"]
   # This normalization is applied per-channel on the 2D data
   for i in range(x.shape[0]):
       the_min, the_max = mins[i], maxs[i]
       if the_max > the_min:
           x[i] = 2 * (x[i] - the_min) / (the_max - the_min) - 1
       else:
           x[i] = 0 # If min == max, set to 0 to avoid division by zero
   return x


def run(split: str, fires: List[Path], out: Path, k: int, sz: Tuple[int,int], stats: Dict):
   """Processes and saves the data for one split using global normalization stats."""
   if not fires: return
   inp_dir, tgt_dir = out/split/"inputs", out/split/"targets"
   inp_dir.mkdir(parents=True, exist_ok=True); tgt_dir.mkdir(parents=True, exist_ok=True)
  
   total = 0
   for fd in tqdm(fires, desc=f"Processing {split} split"):
       for input_sid, imgs, mask_src, target_sid in collect_samples(fd, k):
           try:
               # --- Process Input (Conditions) ---
               arrs = [read_tif(p) for p in imgs] # Read raw TIFFs
               # Resize each image in arrs first
               resized_arrs = [resize(arr, sz, nearest=False) for arr in arrs]
               x = np.concatenate(resized_arrs, 0) # Concatenate channels (C, H, W)
              
               # NEW: Replace NaNs with 0.0 in the input data
               x_processed = np.nan_to_num(x, nan=0.0).astype(np.float32)


               # Apply global normalization (now on data with NaNs replaced)
               x_processed = normalize_globally(x_processed, stats)
              
               np.save(inp_dir/f"{input_sid}.npy", x_processed) # Save processed input (C, H, W)


               # --- Process Target (Mask) ---
               with rasterio.open(mask_src) as src:
                   mask_raw_all_channels = src.read() # Read ALL channels (num_channels, H, W)
              
               # Resize the raw mask (applied to all channels)
               resized_mask_all_channels = resize(mask_raw_all_channels, sz, nearest=True)


               # Initialize a single-channel array for the final processed mask
               # Use float32 to accommodate 0.0 and 255.0
               mask_processed_single_channel = np.zeros(sz, dtype=np.float32)


               # Iterate through each pixel's "stack" of 23 channel values
               # and apply the NaN logic to determine the final single-channel pixel value
               for r in range(resized_mask_all_channels.shape[1]): # Iterate rows
                   for c in range(resized_mask_all_channels.shape[2]): # Iterate columns
                       pixel_values_across_channels = resized_mask_all_channels[:, r, c]
                      
                       # If ANY of these 23 values is NaN, it's "no fire" (0.0).
                       # Otherwise (all values are valid), it's "fire" (255.0).
                       if np.any(np.isnan(pixel_values_across_channels)):
                           mask_processed_single_channel[r, c] = 0.0 # No fire (from NaN)
                       else:
                           mask_processed_single_channel[r, c] = 255.0 # Fire (from valid data)
              
               # Add a channel dimension to the target mask (1, H, W)
               mask_final_target = np.expand_dims(mask_processed_single_channel, axis=0)
              
               np.save(tgt_dir / f"{target_sid}.npy", mask_final_target) # Save final binary target (1, H, W)
               total += 1
           except Exception as e:
               print(f"Skipping sample {input_sid} (target {target_sid}) due to error: {e}")
   print(f"Processed {total} samples for the {split} split.")


def main():
   p = argparse.ArgumentParser()
   p.add_argument("--raw", type=Path, required=True, help="Path to the raw WildfireSpreadTS dataset directory.")
   p.add_argument("--out", type=Path, default=Path("./data_2"), help="Path to the output directory for processed data.")
   p.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
   p.add_argument("--days", type=int, default=3, help="Number of past days to use as conditioning.")
   args = p.parse_args()
  
   stats_path = args.out / "normalization_stats.json"


   fires_by_split = gather_fire_dirs(args.raw)
  
   if not stats_path.exists():
       if not fires_by_split['train']:
           raise RuntimeError("No fire events found in the training split. Cannot calculate global statistics.")
       sample_tif_path = sort_date_tifs(fires[0])[0]
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


