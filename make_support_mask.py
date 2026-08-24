#!/usr/bin/env python
"""Rasterize COCO polygon annotations into a binary support mask for INSID3."""

import argparse
import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: alongside the json)")
    ap.add_argument("--image-id", type=int, default=None, help="which image entry to use")
    args = ap.parse_args()

    with open(args.coco_json) as f:
        data = json.load(f)

    images = data["images"]
    if args.image_id is None:
        img = images[0]
    else:
        img = next(i for i in images if i["id"] == args.image_id)
    h, w = img["height"], img["width"]

    anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        if isinstance(seg, list):                      # polygon(s)
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg["counts"], list):          # uncompressed RLE
            rle = mask_utils.frPyObjects(seg, h, w)
        else:                                          # compressed RLE
            rle = seg
        mask |= mask_utils.decode(rle).astype(np.uint8)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.coco_json)), "support_mask.png")
    Image.fromarray(mask * 255).save(out)

    print(f"annotations : {len(anns)}")
    print(f"resolution  : {w}x{h}")
    print(f"foreground  : {100.0 * mask.mean():.1f}%")
    print(f"reference   : {img['file_name']}")
    print(f"wrote       : {out}")


if __name__ == "__main__":
    main()
