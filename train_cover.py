#!/usr/bin/env python
"""Fit a patch-level residue regressor on frozen (debiased) DINOv3 features.

Each patch gets a continuous target equal to the fraction of its pixels that are
annotated residue. Concatenating several transformer layers keeps finer spatial
detail than the final layer alone, which matters for thin structures.
"""

import argparse
import json

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeCV

from models import build_insid3
from utils.data import load_image


def _intermediate(model, x, n_layers):
    """Run the encoder and return the last `n_layers` feature maps as fp32."""
    if x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fmaps = model.encoder.get_intermediate_layers(x, n=n_layers, reshape=True)
    else:
        fmaps = model.encoder.get_intermediate_layers(x, n=n_layers, reshape=True)
    return [f.float() for f in fmaps]


def build_positional_bases(model, n_layers, svd_components=500):
    """Estimate a positional subspace per layer from a semantically empty image."""
    from torchvision.transforms.functional import normalize
    blank = normalize(
        torch.zeros(1, 3, model.image_size, model.image_size),
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    ).to(model.device)
    bases = []
    for f in _intermediate(model, blank, n_layers):
        f = F.normalize(f, p=2, dim=1)
        E = einops.rearrange(f, "b c h w -> c (b h w)")
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)
        bases.append(U[:, : min(svd_components, U.shape[1])].contiguous())
    return bases


def patch_features(model, path, bases=None, n_layers=1):
    """Return (N, C*n_layers) patch features and the (h, w) feature-grid shape."""
    img, _ = load_image(path, model._transform, model.device)
    cols = []
    for i, f in enumerate(_intermediate(model, img, n_layers)):
        f = F.normalize(f, p=2, dim=1)
        c, h, w = f.shape[1:]
        X = f.reshape(c, -1)
        if bases is not None:
            B = bases[i].to(X.dtype)
            X = X - B @ (B.T @ X)          # project onto orthogonal complement
            X = F.normalize(X, p=2, dim=0)
        cols.append(X)
    return torch.cat(cols, dim=0).T.cpu().numpy(), (h, w)


def patch_targets(mask_path, image_size, hw):
    """Fraction of each patch covered by the annotation, in [0, 1]."""
    m = np.array(Image.open(mask_path).convert("L")) > 127
    t = torch.from_numpy(m).float()[None, None]
    # Match build_transform: resize to a square, then area-average into patches.
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(t, hw)[0, 0].reshape(-1).numpy()


def evaluate(pred, hw, mask_path, thresholds):
    """Pixel-level IoU / precision / recall at each threshold."""
    m = np.array(Image.open(mask_path).convert("L")) > 127
    p = np.asarray(
        Image.fromarray((pred.reshape(hw) * 255).astype(np.uint8))
        .resize((m.shape[1], m.shape[0]), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    print(f"\n{'thresh':>7} {'IoU':>7} {'prec':>7} {'recall':>7} {'pred%':>7}")
    best = (None, -1.0)
    for t in thresholds:
        b = p > t
        inter = float((b & m).sum())
        union = float((b | m).sum())
        iou = inter / max(union, 1.0)
        prec = inter / max(float(b.sum()), 1.0)
        rec = inter / max(float(m.sum()), 1.0)
        print(f"{t:7.2f} {iou:7.3f} {prec:7.3f} {rec:7.3f} {100 * b.mean():7.1f}")
        if iou > best[1]:
            best = (t, iou)
    print(f"best IoU {best[1]:.3f} at threshold {best[0]:.2f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", required=True, action="append")
    ap.add_argument("--ref-mask", required=True, action="append")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--model-size", default="large", choices=["small", "base", "large"])
    ap.add_argument("--layers", type=int, default=1,
                    help="how many trailing transformer layers to concatenate")
    ap.add_argument("--no-debias", action="store_true")
    ap.add_argument("--val-image", action="append", default=[],
                    help="held-out frame for honest evaluation; repeatable")
    ap.add_argument("--val-mask", action="append", default=[])
    ap.add_argument("--out", default="cover_model.npz")
    args = ap.parse_args()

    if len(args.ref_image) != len(args.ref_mask):
        ap.error("--ref-image and --ref-mask must be repeated the same number of times")

    model = build_insid3(model_size=args.model_size, image_size=args.image_size)
    bases = None if args.no_debias else build_positional_bases(model, args.layers)

    xs, ys = [], []
    for img_path, mask_path in zip(args.ref_image, args.ref_mask):
        x, hw = patch_features(model, img_path, bases, args.layers)
        y = patch_targets(mask_path, args.image_size, hw)
        xs.append(x)
        ys.append(y)
        print(f"{img_path}\n  grid={hw[0]}x{hw[1]}  dim={x.shape[1]}  true cover={100 * y.mean():.1f}%")

    X = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)

    reg = RidgeCV(alphas=np.logspace(-2, 4, 13))
    reg.fit(X, y)
    pred = np.clip(reg.predict(X), 0.0, 1.0)

    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    print(f"\nalpha      : {reg.alpha_:.4g}")
    print(f"train R^2  : {1 - ss_res / ss_tot:.3f}")
    print(f"true cover : {100 * y.mean():.2f}%")
    print(f"pred cover : {100 * pred.mean():.2f}%")

    sweep = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # In-sample: optimistic, useful only for comparing configurations.
    n0 = len(ys[0])
    print("\n=== TRAIN (in-sample, optimistic) ===")
    evaluate(pred[:n0], hw, args.ref_mask[0], sweep)

    if len(args.val_image) != len(args.val_mask):
        ap.error("--val-image and --val-mask must be repeated the same number of times")
    for v_img, v_mask in zip(args.val_image, args.val_mask):
        xv, hwv = patch_features(model, v_img, bases, args.layers)
        pv = np.clip(reg.predict(xv.astype(np.float64)), 0.0, 1.0)
        yv = patch_targets(v_mask, args.image_size, hwv)
        print(f"\n=== VAL {v_img} ===")
        print(f"true cover {100 * yv.mean():.2f}%   pred cover {100 * pv.mean():.2f}%")
        evaluate(pv, hwv, v_mask, sweep)

    np.savez(
        args.out,
        coef=reg.coef_,
        intercept=reg.intercept_,
        positional_bases=(np.stack([b.float().cpu().numpy() for b in bases])
                          if bases is not None else np.zeros(0)),
        config=json.dumps({
            "image_size": args.image_size,
            "model_size": args.model_size,
            "layers": args.layers,
            "debias": bases is not None,
        }),
    )
    print(f"wrote      : {args.out}")


if __name__ == "__main__":
    main()
