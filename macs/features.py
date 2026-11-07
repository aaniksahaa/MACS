"""Pooled deep features of any teacher architecture, with a small on-disk cache."""
import os

import numpy as np
import torch

from .utils import file_tag, make_loader


@torch.no_grad()
def pooled_features(model, x):
    """Global max-pool of the deepest encoder map: [B, C]."""
    name = type(model).__name__
    if name == "UNet":
        for down in model.downs:
            x = model.pool(down(x))
        x = model.bottleneck(x)
    elif name == "ResUNet":
        x = model.conv_in(x)
        for layer in model.down_layers:
            x = layer(x)
    elif name == "DeepLabV3":
        for layer in model.encoder:
            x = layer(x)
    else:
        raise TypeError(f"no feature hook for architecture {name}")
    return torch.amax(x, dim=(2, 3))


@torch.no_grad()
def extract(model, dataset, indices, device, batch_size=16, num_workers=8):
    model.eval()
    out = []
    for imgs, _ in make_loader(dataset, indices, batch_size, num_workers):
        out.append(pooled_features(model, imgs.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


class FeatureStore:
    """cache/<dir>/features/<data>__<split>__n<N>__s<seed>__via_<arch>-<encoder>__<ckpt tag>.npz"""

    def __init__(self, cache_dir, enabled=True):
        self.dir = os.path.join(cache_dir, "features")
        self.enabled = enabled
        if enabled:
            os.makedirs(self.dir, exist_ok=True)

    def path(self, data_domain, split, n, seed, encoder_domain, arch, ckpt):
        return os.path.join(self.dir, f"{data_domain}__{split}__n{n}__s{seed}__via_{arch}-{encoder_domain}__{file_tag(ckpt)}.npz")

    def get(self, data_domain, split, n, seed, encoder_domain, arch, ckpt, compute):
        """`compute()` is called on a miss; returns float32 [n_actual, C]."""
        path = self.path(data_domain, split, n, seed, encoder_domain, arch, ckpt)
        if self.enabled and os.path.isfile(path):
            arr = np.load(path)["features"]
            if np.isfinite(arr).all() and arr.ndim == 2:
                return arr, True
        arr = np.asarray(compute(), dtype=np.float32)
        if self.enabled:
            tmp = path + ".tmp.npz"
            np.savez(tmp, features=arr)
            os.replace(tmp, path)
        return arr, False
