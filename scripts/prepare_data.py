from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

#rasterio is a useful library you can use to basically work with geospatial data (hence why we're using it now)
import rasterio
from PIL import Image


#The code here assumes that WildFireSpreadTS's folder layout is such that all the data is within the "train" folder btw


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

#essenitally just normalizing all channels to a [-1, 1] clamp
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


#this will collect valid pairs of input/output for a time series task, all from the directory
#by iterating through directories in base, and then identifying different tiff files based on different days, we can sequences of k days, while also checking the day's file still exists

def collect_samples(base, k):
    out = []
    for folder in base.iterdir():
        if not folder.is_dir(): 
            continue
        files = sorted(folder.glob("day_*.tif"))
        days = {int(f.stem.split("_")[-1]): f for f in files}
        max_d = max(days.keys())
        for t in range(k-1, max_d):
            try:
                frames = [days[t-i] for i in reversed(range(k))]
            except KeyError:
                continue
            msk = folder / f"day_{t+1:03d}_mask.tif"
            if msk.exists():
                sid = f"{folder.name}_d{t:03d}"
                out.append((sid, frames, msk))
    return out



#decided to split the data in this file HERE (so training / validation / test)
def run(split, src, out, k, sz, do_norm):
    print(f"Split: {split}")
    inp = out / split / "inputs"
    tgt = out / split / "targets"
    inp.mkdir(parents=True, exist_ok=True)
    tgt.mkdir(parents=True, exist_ok=True)
    data = collect_samples(src / split, k)
    print(f"{len(data)} samples found")
    for sid, imgs, m in data:
        arrs = [resize(read_tif(p), sz) for p in imgs]
        x = np.concatenate(arrs, axis=0)
        if do_norm:
            x = normalize(x)
        np.save(inp / f"{sid}.npy", x)
        msk = read_tif(m)[0]
        msk = resize(msk[None], sz, nearest=True)[0]
        msk = (msk > 0).astype(np.uint8) * 255
        Image.fromarray(msk).save(tgt / f"{sid}.png")

#some parsing, had to ask gpt for some of this code in this function
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./data"))
    p.add_argument("--image-size", type=int, nargs=2, default=(64, 64))
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--no-normalise", action="store_true")
    args = p.parse_args()

    for sp in ["train", "val", "test"]:
        run(sp, args.raw, args.out, args.days, tuple(args.image_size), not args.no_normalise)

if __name__ == "__main__":
    main()
