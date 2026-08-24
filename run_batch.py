#!/usr/bin/env python
"""Segment a directory of images with INSID3 from a single reference image + mask."""

import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

from models import build_insid3

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", required=True, action="append",
                    help="reference image; repeat for multiple supports")
    ap.add_argument("--ref-mask", required=True, action="append",
                    help="reference mask; repeat, paired with --ref-image")
    ap.add_argument("--combine", default="union", choices=["union", "multishot"],
                    help="union: run each support separately and OR the masks (better for "
                         "visually heterogeneous classes). multishot: stack supports into one "
                         "averaged prototype.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--merge-thresh", type=float, default=0.2)
    ap.add_argument("--model-size", default="large", choices=["small", "base", "large"])
    ap.add_argument("--crf", action="store_true", help="CRF mask refinement instead of bilinear")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images")
    ap.add_argument("--overlay", action="store_true", help="also write a visual overlay JPG")
    args = ap.parse_args()

    if len(args.ref_image) != len(args.ref_mask):
        ap.error("--ref-image and --ref-mask must be repeated the same number of times")
    supports = list(zip(args.ref_image, args.ref_mask))

    pattern = "**/*" if args.recursive else "*"
    files = [
        p for p in sorted(glob.glob(os.path.join(args.input_dir, pattern), recursive=args.recursive))
        if p.lower().endswith(EXTS)
    ]
    ref_abs = {os.path.abspath(img) for img, _ in supports}
    files = [p for p in files if os.path.abspath(p) not in ref_abs]
    if args.limit:
        files = files[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"found {len(files)} images in {args.input_dir}", flush=True)

    model = build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        mask_refiner="crf" if args.crf else "bilinear",
    )
    start = time.time()
    done = 0
    for idx, path in enumerate(files, 1):
        rel = os.path.relpath(path, args.input_dir)
        stem = os.path.splitext(rel)[0].replace(os.sep, "__")
        out_path = os.path.join(args.output_dir, stem + "_mask.png")
        if os.path.exists(out_path):          # resume-safe
            continue

        # segment() calls reset_state(), which clears the references too, so they
        # must be re-set for every target.
        if args.combine == "multishot":
            for ref_img, ref_msk in supports:
                model.set_reference(ref_img, ref_msk)
            model.set_target(path)
            mask = model.segment().cpu().numpy().astype(bool)
        else:
            mask = None
            for ref_img, ref_msk in supports:
                model.set_reference(ref_img, ref_msk)
                model.set_target(path)
                one = model.segment().cpu().numpy().astype(bool)
                mask = one if mask is None else (mask | one)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

        if args.overlay:
            rgb = np.array(Image.open(path).convert("RGB"))
            if rgb.shape[:2] == mask.shape:
                rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
                Image.fromarray(rgb).save(os.path.join(args.output_dir, stem + "_overlay.jpg"), quality=88)

        done += 1
        rate = (time.time() - start) / done
        print(f"[{idx}/{len(files)}] {rel}  fg={100.0 * mask.mean():.1f}%  {rate:.2f}s/img", flush=True)

    print(f"finished: {done} newly written, output in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
