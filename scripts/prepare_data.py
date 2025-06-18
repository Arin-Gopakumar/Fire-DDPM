from __future__ import annotations

import argparse, json, re
import os
from pathlib import Path
import numpy as np

#rasterio is a useful library you can use to basically work with geospatial data (hence why we're using it now)
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from typing import Dict, List, Tuple


#The code here assumes that WildFireSpreadTS's folder layout is such that all the data is within the "train" folder btw

#this here is regular expression (RegEx), used to define patterns of a date format, like "2019-04-19"
date = re.compile(r"(\d{4}-\d{2}-\d{2})")

#here, we're resizing every channel of our image to a new shape
def resize(img, shape, nearest=False):
    res = np.empty((img.shape[0], *shape), dtype=img.dtype)
    #iterate through the channel dimension, and use im.resize, and store that in a new array
    for i in range(img.shape[0]):
        im = Image.fromarray(img[i])
        #keep in mind we're using nearest neighbor interpolation if nearest is True
        method = Image.NEAREST if nearest else Image.BILINEAR
        res[i] = np.asarray(im.resize(shape[::-1], method))
    return res

#essentially just normalizing all channels to a [-1, 1] clamp
def normalize(x):
    x = x.astype(np.float32)
    for i in range(x.shape[0]):
        the_min = x[i].min()
        the_max = x[i].max()
        if the_max != the_min:
            x[i] = 2 * (x[i] - the_min) / (the_max - the_min) - 1
        else:
            x[i] = 0
    return x


#this will read a tif file and return a numpy array, using rasterio
def read_tif(p):
    with rasterio.open(p) as f:
        return f.read()

def year_to_split(year):
    #feel free to move around these values as needed
    if year <= 2019:
        return "train"
    if year == 2020:
        return "val"
    return "test"

def gather_fire_dirs(root, split_map):
    fires = {"train": [], "val": [], "test": []}
    for year_dir in root.glob("[12][0-9][0-9][0-9]"):
        if not year_dir.is_dir():
            continue
        sp = split_map.get(year_dir.name, year_to_split(int(year_dir.name)))
        for fd in year_dir.iterdir():
            if fd.is_dir():
                fires[sp].append(fd)
    return fires

def sort_date_tifs(fire_dir):
    return sorted([p for p in fire_dir.glob("*.tif") if date.search(p.stem)],
                  key=lambda p: date.search(p.stem).group(1))



def collect_samples(fire_dir, k):
    #Return (sample_id, [t-k+1..t], mask_{t+1}) for one fire
    tifs = sort_date_tifs(fire_dir)
    if len(tifs) < k+1:
        return []
    samples = []
    for idx in range(k-1, len(tifs)-1):
        past = tifs[idx-k+1: idx+1]
        nxt = tifs[idx+1]
        sid = f"{fire_dir.name}_{date.search(past[-1].stem).group(1)}"
        samples.append((sid, past, nxt))
    return samples



#decided to split the data in this file HERE (so training / validation / test)
# prepares data for one split
#for each pair, you load and resize the tif stack, normalize if needed, save it as an .npy file, binarizes the mask file, and then finally saves it as a png
def run(split: str, fires: List[Path], out: Path, k: int, sz: Tuple[int,int], norm: bool):
    if not fires:
        return
    inp_dir, tgt_dir = out/split/"inputs", out/split/"targets"
    inp_dir.mkdir(parents=True, exist_ok=True); tgt_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for fd in fires:
        for sid, imgs, mask_src in collect_samples(fd, k):
            # stack k days
            arrs = [resize(read_tif(p), sz, nearest=False) for p in imgs]
            x = np.concatenate(arrs, 0)
            if norm:
                x = normalize(x)
            np.save(inp_dir/f"{sid}.npy", x)

            # mask: try last band == binary else sibling _mask.tif
            m_arr = None
            with rio.open(mask_src) as ds:
                if ds.count >= 2:
                    lb = ds.read(ds.count)
                    if lb.max() <= 1 and lb.dtype in (np.uint8, np.uint16, np.bool_):
                        m_arr = lb
            if m_arr is None:
                alt = mask_src.with_name(f"{mask_src.stem}_mask{mask_src.suffix}")
                if alt.exists():
                    m_arr = read_tif(alt)[0]
            if m_arr is None:
                continue  # skip if no mask
            m_arr = resize(m_arr[None], sz, nearest=True)[0]
            m_arr = (m_arr>0).astype(np.uint8)*255
            Image.fromarray(m_arr).save(tgt_dir/f"{sid}.png")
            total += 1

#some parsing, had to ask gpt for some of this code in this function
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./data"))
    p.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--no-normalise", action="store_true")
    p.add_argument("--split-map", type=Path, help="optional, for JSON year to split map")
    args = p.parse_args()

    mapping = json.loads(args.split_map.read_text()) if args.split_map else {}
    fires_by_split = gather_fire_dirs(args.raw, mapping)
    for sp in ["train","val","test"]:
        run(sp, fires_by_split[sp], args.out, args.days, tuple(args.image_size), not args.no_normalise)

if __name__ == "__main__":
    main()
