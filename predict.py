#!/usr/bin/env python3
"""Segment EM images with a trained model (a MACS student or a source model).

    # a split of a domain, with metrics against its labels (the evaluation of train_macs.py)
    python predict.py --checkpoint outputs/membrane/snemi3d/b10_s1/model.pth --task membrane --domain snemi3d --out preds/snemi3d
    # any folder of grayscale images, no labels needed
    python predict.py --checkpoint outputs/membrane/snemi3d/b10_s1/model.pth --images my_sections/ --out preds/mine

Images are tiled into 256x256 patches (stride 128), overlapping predictions are averaged, and every image gets
<stem>_prob.png (uint8, 255 x foreground probability) and <stem>_mask.png (0/255, probability > 0.5).
With --domain, metrics.csv holds the per-image metrics and the summary is printed (patch and image level, as in
result.json). Test-time augmentation (8 flips / rotations) is on by default, as for the reported MACS results.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from archs import load_model
from common.data import _open_gray
from macs import utils
from macs.evaluate import evaluate_model, tta_probs

PATCH, STRIDE = utils.PATCH_SIZE, utils.STRIDE
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def load(checkpoint, device):
    """MACS students are plain U-Net state_dicts; source models carry a .meta.json with their architecture."""
    model, _ = load_model(checkpoint, device)
    return model.eval()


def save_maps(out_dir, stem, prob):
    Image.fromarray((np.clip(prob, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(out_dir, f"{stem}_prob.png"))
    Image.fromarray(((prob > 0.5) * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{stem}_mask.png"))


def positions(size):
    """Patch origins along one axis that cover the whole axis (the last patch is flush with the border)."""
    last = size - PATCH
    pos = list(range(0, last + 1, STRIDE))
    if pos[-1] != last:
        pos.append(last)
    return pos


@torch.no_grad()
def predict_image(model, array, device, tta, amp, batch_size):
    """Foreground probability map of one grayscale image (uint8 array), the same size as the image."""
    h0, w0 = array.shape
    pad_b, pad_r = max(0, PATCH - h0), max(0, PATCH - w0)
    if pad_b or pad_r:
        array = np.pad(array, ((0, pad_b), (0, pad_r)), mode="reflect")
    h, w = array.shape
    image = torch.from_numpy(array.astype(np.float32) / 255.0)
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    origins = [(y, x) for y in positions(h) for x in positions(w)]
    for i in range(0, len(origins), batch_size):
        chunk = origins[i:i + batch_size]
        batch = torch.stack([image[y:y + PATCH, x:x + PATCH] for y, x in chunk]).unsqueeze(1).to(device)
        with torch.autocast(device_type="cuda", enabled=amp and str(device).startswith("cuda")):
            probs = tta_probs(model, batch) if tta else torch.sigmoid(model(batch).float())
        probs = probs.float().cpu().numpy()[:, 0]
        for (y, x), p in zip(chunk, probs):
            acc[y:y + PATCH, x:x + PATCH] += p
            cnt[y:y + PATCH, x:x + PATCH] += 1
    return (acc / cnt)[:h0, :w0]


def predict_folder(model, args, device):
    names = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMAGE_SUFFIXES))
    for n, name in enumerate(names, 1):
        array = np.asarray(_open_gray(os.path.join(args.images, name)))
        prob = predict_image(model, array, device, args.tta, args.amp, args.batch_size)
        save_maps(args.out, os.path.splitext(name)[0], prob)
        print(f"[{n}/{len(names)}] {name}: foreground {float((prob > 0.5).mean()):.3f}", flush=True)


def predict_domain(model, args, device):
    """The evaluation of train_macs.py (same patch grid, stitching and metrics), keeping the stitched maps."""
    utils.set_task(args.task, args.data_root, args.label_dir)
    ds = utils.make_dataset(args.domain, args.split)
    rows = []

    def on_image(img_idx, prob, covered, mask, metrics):
        name = ds.filenames[img_idx]
        stem = os.path.splitext(name)[0]
        save_maps(args.out, stem, prob)
        rows.append({"image": name, "covered_pixels": int(covered.sum()), **metrics})

    result = evaluate_model(model, ds, device, args.batch_size, args.num_workers, image_level=True, amp=args.amp,
                            tta=args.tta, on_image=on_image)
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    utils.save_json(os.path.join(args.out, "summary.json"), {"checkpoint": os.path.abspath(args.checkpoint),
                                                            "domain": args.domain, "split": args.split, "tta": args.tta,
                                                            **result})
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model.pth of a MACS run, or a source model checkpoint")
    ap.add_argument("--out", required=True, help="output folder for the probability maps and masks")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of grayscale images (no labels needed)")
    src.add_argument("--domain", help="a domain of --task (or a sub-folder of --data-root); predicts --split with metrics")
    ap.add_argument("--task", default="membrane")
    ap.add_argument("--data-root", default="", help="folder with one sub-folder per domain (default: the registry)")
    ap.add_argument("--label-dir", default="", help="label folder name (default: the task's)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model = load(args.checkpoint, device)
    if args.images:
        predict_folder(model, args, device)
    else:
        predict_domain(model, args, device)


if __name__ == "__main__":
    main()
