"""Patch-based binary segmentation dataset shared by train_source.py and macs/.

One class serves every task: the label folder (``membranes``, ``mitochondria``, ...)
is a parameter, see ``common/domains.py``.
"""
import functools
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Decoded grayscale images are kept in a per-process LRU cache, so a patch does not decode its whole parent
# image again; MACS_IMAGE_CACHE sets its size (images per DataLoader worker, default 256, about 1 MB each for
# 1024x1024 sections; 0 disables it).
_IMAGE_CACHE_SIZE = int(os.environ.get("MACS_IMAGE_CACHE", "256"))


@functools.lru_cache(maxsize=_IMAGE_CACHE_SIZE if _IMAGE_CACHE_SIZE > 0 else None)
def _load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L")).copy()


def _open_gray(path):
    array = _load_gray(path) if _IMAGE_CACHE_SIZE > 0 else _load_gray.__wrapped__(path)
    return Image.fromarray(array)


class PatchSegDataset(Dataset):
    """Sliding-window patches over ``<root_dir>/<split>/raw`` and ``.../<label_dir>``.

    Images are read as grayscale; a patch grid with ``patch_size`` and ``stride``
    is laid over every image (no padding, so border remainders are dropped).
    ``binarize`` thresholds the mask at 0.5, as training and MACS always did;
    MACS evaluation keeps the mask tensor untouched (``binarize=False``).

    Two dataset quirks are handled here so that every domain goes through the
    same pipeline:

    * an image smaller than ``patch_size`` in either direction is reflect-padded
      (image and mask alike) up to ``patch_size``, giving it one patch per axis;
      images at least ``patch_size`` wide and high are never touched;
    * ``ignore_label`` names a mask value that marks "no tissue" (padding outside
      the imaged volume). Patches whose mask is entirely that value are dropped
      from the grid; elsewhere the value falls below the 0.5 threshold and counts
      as background.
    """

    def __init__(self, root_dir, split, patch_size=256, stride=128, transform=None,
                 binarize=True, verbose=True, label_dir="membranes", ignore_label=None):
        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.mask_dir = os.path.join(root_dir, split, label_dir)
        self.label_dir = label_dir
        if not os.path.isdir(self.raw_dir):
            raise FileNotFoundError(f"Missing raw directory: {self.raw_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Missing label directory: {self.mask_dir}")
        self.filenames = sorted(os.listdir(self.raw_dir))
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.binarize = binarize
        self.ignore_label = ignore_label
        self.padded = {}            # img_idx -> (pad_right, pad_bottom) for undersized images
        self.patches = []
        dropped = 0
        for img_idx, fname in enumerate(self.filenames):
            with Image.open(os.path.join(self.raw_dir, fname)) as img:
                w, h = img.size
            if w < patch_size or h < patch_size:
                self.padded[img_idx] = (max(0, patch_size - w), max(0, patch_size - h))
                w, h = max(w, patch_size), max(h, patch_size)
            mask = None
            if ignore_label is not None:
                mask = np.asarray(self._open_mask(fname))
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    if mask is not None and np.all(mask[y:y + patch_size, x:x + patch_size] == ignore_label):
                        dropped += 1
                        continue
                    self.patches.append((img_idx, y, x))
        if verbose:
            extra = []
            if self.padded:
                extra.append(f"{len(self.padded)} image(s) reflect-padded to {patch_size}")
            if dropped:
                extra.append(f"{dropped} patch(es) of ignore label {ignore_label} only dropped")
            suffix = f" ({'; '.join(extra)})" if extra else ""
            print(f"[{split}] Total patches: {len(self.patches)}{suffix}")

    def _pad(self, image, img_idx):
        pad_right, pad_bottom = self.padded[img_idx]
        array = np.pad(np.asarray(image), ((0, pad_bottom), (0, pad_right)), mode="reflect")
        return Image.fromarray(array)

    def _open_mask(self, fname, img_idx=None):
        mask = _open_gray(os.path.join(self.mask_dir, fname))
        if img_idx in self.padded:
            mask = self._pad(mask, img_idx)
        return mask

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        img = _open_gray(os.path.join(self.raw_dir, fname))
        if img_idx in self.padded:
            img = self._pad(img, img_idx)
        mask = self._open_mask(fname, img_idx)
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        if self.transform:
            # Same RNG state for image and mask so paired random augmentations agree.
            seed = torch.seed()
            torch.manual_seed(seed)
            img_patch = self.transform(img_patch)
            torch.manual_seed(seed)
            mask_patch = T.ToTensor()(mask_patch)
        else:
            img_patch = T.ToTensor()(img_patch)
            mask_patch = T.ToTensor()(mask_patch)

        if self.binarize:
            mask_patch = (mask_patch > 0.5).float()

        return img_patch, mask_patch
