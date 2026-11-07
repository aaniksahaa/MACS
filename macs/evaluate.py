"""Patch-level metrics, image-level metrics after stitching overlapping patches, calibration."""
import numpy as np
import torch

from .calibration import calibration_metrics
from .utils import make_loader


def counts(pred_bin, mask):
    pred_bin = pred_bin.astype(bool)
    mask = mask.astype(bool)
    tp = int(np.logical_and(pred_bin, mask).sum())
    fp = int(np.logical_and(pred_bin, ~mask).sum())
    fn = int(np.logical_and(~pred_bin, mask).sum())
    tn = int(pred_bin.size - tp - fp - fn)
    return tp, fp, fn, tn


def vi_from_counts(tp, fp, fn, tn, eps=1e-10):
    """Binary-mask variation of information (natural log)."""
    total = float(tp + fp + fn + tn)
    pxy = np.array([[tn, fp], [fn, tp]], dtype=np.float64) / total   # rows: true, cols: pred
    px, py = pxy.sum(1), pxy.sum(0)

    def h(p):
        p = p + eps
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    ixy = float(np.nansum(pxy * np.log((pxy + eps) / (px[:, None] * py[None, :] + eps))))
    return h(px) + h(py) - 2 * ixy


def metrics_from_counts(tp, fp, fn, tn):
    dice = (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "vi": vi_from_counts(tp, fp, fn, tn)}


@torch.no_grad()
def patch_dice(probs, masks, smooth=1.0):
    pred = (probs > 0.5).float()
    inter = (pred * masks).sum(dim=(1, 2, 3))
    return (2 * inter + smooth) / (pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth)


class ImageStitcher:
    """Accumulates overlapping patch probabilities per test image; metrics on the stitched map.
    on_image(img_idx, prob, covered, mask, metrics), if given, receives every finished image (full-size maps)."""

    def __init__(self, dataset, on_image=None):
        self.dataset = dataset
        self.on_image = on_image
        self.current = None
        self.records = []

    def _flush(self):
        if self.current is None:
            return
        img_idx, acc, cnt = self.current
        covered = cnt > 0
        prob = acc[covered] / cnt[covered]
        mask = np.asarray(self.dataset._open_mask(self.dataset.filenames[img_idx], img_idx)) > 127
        self.records.append(metrics_from_counts(*counts(prob > 0.5, mask[covered])))
        if self.on_image is not None:
            full = np.zeros(acc.shape, np.float32)
            full[covered] = prob
            self.on_image(img_idx, full, covered, mask, self.records[-1])
        self.current = None

    def add(self, patch_index, prob):
        img_idx, y, x = self.dataset.patches[patch_index]
        if self.current is None or self.current[0] != img_idx:
            self._flush()
            with_pad = self.dataset.padded.get(img_idx)
            from PIL import Image
            import os
            with Image.open(os.path.join(self.dataset.raw_dir, self.dataset.filenames[img_idx])) as im:
                w, h = im.size
            if with_pad:
                w, h = max(w, self.dataset.patch_size), max(h, self.dataset.patch_size)
            self.current = (img_idx, np.zeros((h, w), np.float32), np.zeros((h, w), np.int32))
        _, acc, cnt = self.current
        s = self.dataset.patch_size
        acc[y:y + s, x:x + s] += prob
        cnt[y:y + s, x:x + s] += 1

    def result(self):
        self._flush()
        if not self.records:
            return {}
        return {f"image_{k}": float(np.mean([r[k] for r in self.records])) for k in self.records[0]} | {
            "n_images": len(self.records)}


def _dihedral(x, k, inverse=False):
    if inverse and (k & 4):
        x = x.transpose(-1, -2)
    if k & 2:
        x = x.flip(-2)
    if k & 1:
        x = x.flip(-1)
    if not inverse and (k & 4):
        x = x.transpose(-1, -2)
    return x


@torch.no_grad()
def tta_probs(model, imgs):
    """Mean probability over the 8 flips / rotations (test-time augmentation)."""
    acc = None
    for k in range(8):
        p = torch.sigmoid(model(_dihedral(imgs, k)).float())
        p = _dihedral(p, k, inverse=True)
        acc = p if acc is None else acc + p
    return acc / 8


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=16, num_workers=8, image_level=True,
                   calibration_stride=8, amp=True, tta=False, on_image=None):
    """Patch-level means, image-level means (stitched), pixel calibration of the model."""
    model.eval()
    stitcher = ImageStitcher(dataset, on_image) if image_level else None
    patch_records, probs_sample, label_sample = [], [], []
    index = 0
    loader = make_loader(dataset, None, batch_size, num_workers)
    for imgs, masks in loader:
        with torch.autocast(device_type="cuda", enabled=amp and str(device).startswith("cuda")):
            x = imgs.to(device, non_blocking=True)
            probs = tta_probs(model, x) if tta else torch.sigmoid(model(x).float())
        probs = probs.cpu().numpy()[:, 0]
        masks = masks.numpy()[:, 0]
        for p, m in zip(probs, masks):
            patch_records.append(metrics_from_counts(*counts(p > 0.5, m > 0.5)))
            if stitcher is not None:
                stitcher.add(index, p)
            index += 1
        probs_sample.append(probs[:, ::calibration_stride, ::calibration_stride].reshape(-1))
        label_sample.append(masks[:, ::calibration_stride, ::calibration_stride].reshape(-1))
    out = {f"patch_{k}": float(np.mean([r[k] for r in patch_records])) for k in patch_records[0]}
    out["n_patches"] = len(patch_records)
    if stitcher is not None:
        out.update(stitcher.result())
    cal = calibration_metrics(np.concatenate(probs_sample), np.concatenate(label_sample))
    out.update({f"cal_{k}": v for k, v in cal.items()})
    return out
