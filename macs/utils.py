"""Shared helpers: registry-based data access, deterministic sampling, teacher loading, logging."""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common import domains as registry  # noqa: E402
from common.data import PatchSegDataset  # noqa: E402
from archs import load_model  # noqa: E402

PATCH_SIZE = 256
STRIDE = 128


TASK = "membrane"                      # set once per process by set_task(); every default below follows it
DATA_ROOT = ""                         # non-empty: domains are the sub-folders of this folder instead of the registry
LABEL_DIR = ""                         # non-empty: label folder name instead of the task's


def set_task(task, data_root="", label_dir=""):
    global TASK, DATA_ROOT, LABEL_DIR
    if task not in registry.TASKS:
        raise SystemExit(f"unknown task {task}; known: {', '.join(registry.TASKS)}")
    TASK, DATA_ROOT, LABEL_DIR = task, data_root, label_dir


def task_domains(task=None):
    """Domains of the current task (by default the sources of a target are the task's other domains)."""
    if DATA_ROOT:
        return sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d, "train")))
    return list(registry.TASKS[task or TASK].domains)


def canonical(domain, task=None):
    if DATA_ROOT:
        return registry.canonical_domain(domain)
    return registry.resolve(domain, task or TASK)[2].name


class _Short(dict):
    def __missing__(self, key):            # mitochondria domains and anything new: the name itself, shortened
        return key[:12]


SHORT = _Short({
    "snemi3d": "snemi", "microns_basil": "microns", "dhanyasi-P14-mouse-cerebellum": "cerebellum",
    "whole-mouse-brain": "mouse-brain", "flywire": "flywire", "c-elegans-dauer-stage": "c-elegans",
    "katz-lab-berghia-connective": "berghia", "octopus-vulgaris-vertical-lobe-glia-deep-neuropil": "oct-glia",
    "octopus-vulgaris-vertical-lobe-sfltract": "oct-sfl",
    "fly-brain-mito-benchmark": "fly-mito", "kasthuri-pp": "kasthuri", "lucchi-pp": "lucchi",
    "drosophila-vnc": "dros-vnc", "multiclass": "multiclass",
})


class Logger:
    def __init__(self, path=None):
        self.file = open(path, "a", buffering=1) if path else None
        self.t0 = time.time()

    def __call__(self, stage, message):
        line = f"[{time.strftime('%H:%M:%S')}][{time.time() - self.t0:7.0f}s][{stage}] {message}"
        print(line, flush=True)
        if self.file:
            self.file.write(line + "\n")

    def close(self):
        if self.file:
            self.file.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def sample_indices(count, wanted, *seed_parts):
    """Deterministic random subset (sorted) of range(count) with at most `wanted` elements."""
    if wanted is None or wanted >= count:
        return np.arange(count)
    rng = np.random.RandomState(stable_seed(*seed_parts))
    return np.sort(rng.choice(count, size=wanted, replace=False))


def make_dataset(domain, split, task=None, binarize=True, verbose=False):
    task = task or TASK
    name = canonical(domain, task)
    path = registry.domain_path(name, task, DATA_ROOT or None)
    return PatchSegDataset(path, split, patch_size=PATCH_SIZE, stride=STRIDE, transform=None,
                           binarize=binarize, verbose=verbose, label_dir=LABEL_DIR or registry.label_dir(task))


def make_loader(dataset, indices=None, batch_size=16, num_workers=8, shuffle=False):
    ds = dataset if indices is None else Subset(dataset, [int(i) for i in indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, persistent_workers=False)


def teacher_checkpoint(models_dir, domain, arch, which="model_best.pth", task=None):
    task = task or TASK
    path = os.path.join(registry.model_dir(models_dir, task, canonical(domain, task), arch, allow_unknown=bool(DATA_ROOT)),
                        which)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing teacher checkpoint: {path}")
    return path


def load_teacher(models_dir, domain, arch, device, which="model_best.pth"):
    path = teacher_checkpoint(models_dir, domain, arch, which)
    model, config = load_model(path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, path


def file_tag(path):
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def save_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(value, fh, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def weight_stats(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    nz = alpha[alpha > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {"max_weight": float(alpha.max()), "entropy": entropy,
            "effective_sources": float(np.exp(entropy)), "min_weight": float(alpha.min())}


class Resources:
    """Per-phase wall time, peak GPU memory and peak CPU RSS, kept as a reproducibility artifact."""

    def __init__(self, device):
        self.device = device
        self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.phases = {}
        self._start = None

    @staticmethod
    def rss_mb():
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def start(self, name):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = (name, time.time())

    def stop(self, extra=None):
        name, t0 = self._start
        if self.cuda:
            torch.cuda.synchronize()
        rec = {"seconds": round(time.time() - t0, 2), "peak_cpu_rss_mb": round(self.rss_mb(), 1)}
        if self.cuda:
            rec["peak_gpu_alloc_mb"] = round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
            rec["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved(self.device) / 2**20, 1)
        if extra:
            rec.update(extra)
        self.phases[name] = rec
        return rec

    def summary(self):
        peak = max([p.get("peak_gpu_alloc_mb", 0) for p in self.phases.values()] + [0])
        return {"phases": self.phases, "peak_gpu_alloc_mb": peak, "peak_cpu_rss_mb": round(self.rss_mb(), 1),
                "total_seconds": round(sum(p["seconds"] for p in self.phases.values()), 1)}


def environment():
    import platform
    import subprocess
    info = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "argv": sys.argv, "cwd": os.getcwd()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        info["commit"] = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
        info["dirty"] = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "macs", "common", "archs"], text=True).strip())
    except Exception:
        pass
    return info
