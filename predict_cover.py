#!/usr/bin/env python
"""Apply a trained patch-level cover regressor to a directory of images."""

import argparse
import csv
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import cm
from PIL import Image

from models import build_insid3
from train_cover import patch_features

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cover_model.npz")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth frame; use to sample across a long clip")
    ap.add_argument("--heatmap-dir", default=None, help="raw per-patch cover heatmaps")
    ap.add_argument("--overlay-dir", default=None,
                    help="colour-mapped cover overlaid on the original frame")
    ap.add_argument("--threshold", type=float, default=None,
                    help="also write a binary overlay at this predicted-fraction cutoff")
    args = ap.parse_args()

    blob = np.load(args.model, allow_pickle=True)
    cfg = json.loads(str(blob["config"]))
    coef, intercept = blob["coef"], float(blob["intercept"])
    print(f"model: image_size={cfg['image_size']} model_size={cfg['model_size']} debias={cfg['debias']}")

    pattern = "**/*" if args.recursive else "*"
    files = [
        p for p in sorted(glob.glob(os.path.join(args.input_dir, pattern), recursive=args.recursive))
        if p.lower().endswith(EXTS)
    ]
    if args.stride > 1:
        files = files[:: args.stride]
    if args.limit:
        files = files[: args.limit]
    print(f"found {len(files)} images")

    model = build_insid3(model_size=cfg["model_size"], image_size=cfg["image_size"])

    import torch
    stored = blob["positional_bases"] if "positional_bases" in blob else np.zeros(0)
    if stored.size:
        bases = [torch.from_numpy(b).to(model.device) for b in stored]
        print(f"using {len(bases)} positional bases stored at training time")
    else:
        bases = None
        if cfg.get("debias"):
            print("WARNING: model predates basis persistence; retrain for reproducible features")
    n_layers = cfg.get("layers", 1)
    for d in (args.heatmap_dir, args.overlay_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    start = time.time()
    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "percent_cover"])
        for idx, path in enumerate(files, 1):
            x, hw = patch_features(model, path, bases, n_layers)
            pred = np.clip(x @ coef + intercept, 0.0, 1.0)
            cover = 100.0 * float(pred.mean())

            rel = os.path.relpath(path, args.input_dir)
            writer.writerow([rel, f"{cover:.2f}"])
            fh.flush()

            stem = os.path.splitext(rel)[0].replace(os.sep, "__")
            if args.heatmap_dir or args.overlay_dir:
                rgb = Image.open(path).convert("RGB")
                # The feature grid comes from a square resize, so stretch it back to
                # the original frame's aspect ratio before overlaying.
                prob = np.asarray(
                    Image.fromarray((pred.reshape(hw) * 255).astype(np.uint8))
                    .resize(rgb.size, Image.BILINEAR)
                ).astype(np.float32) / 255.0
                base = np.asarray(rgb).astype(np.float32)

            if args.heatmap_dir:
                Image.fromarray((prob * 255).astype(np.uint8)).save(
                    os.path.join(args.heatmap_dir, stem + "_cover.png"))

            if args.overlay_dir:
                colour = (cm.turbo(prob)[..., :3] * 255).astype(np.float32)
                Image.fromarray((0.55 * base + 0.45 * colour).astype(np.uint8)).save(
                    os.path.join(args.overlay_dir, stem + "_cover.jpg"), quality=88)

                if args.threshold is not None:
                    hit = prob > args.threshold
                    binv = base.copy()
                    binv[hit] = 0.45 * binv[hit] + 0.55 * np.array([255.0, 40.0, 40.0])
                    Image.fromarray(binv.astype(np.uint8)).save(
                        os.path.join(args.overlay_dir, stem + f"_thresh{args.threshold:g}.jpg"),
                        quality=88)

            print(f"[{idx}/{len(files)}] {rel}  cover={cover:.1f}%  {(time.time() - start) / idx:.2f}s/img",
                  flush=True)

    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
