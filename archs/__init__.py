"""Segmentation architectures used in MACS, behind one small registry.

    from archs import build_model, load_model, list_models
    model = build_model("resunet", features=[32, 64, 128, 256, 512], act_mode="elu")
    model, config = load_model("models/membrane_domains/resunet/snemi3d/model_best.pth", device)

Every architecture takes a grayscale patch and returns one logit map at input
resolution, so train_source.py and macs/ treat them uniformly. ``unet`` is the
MACS student and the default source model; ``resunet`` and ``deeplabv3`` are the
alternative source architectures (heterogeneous teacher pools).

A trained checkpoint is a plain ``state_dict`` (what MACS loads) plus a sidecar
``<checkpoint>.meta.json`` holding the constructor configuration, so
``load_model`` can rebuild the right architecture without being told which one.
"""
import inspect
import json
import os

import torch

from . import blocks
from .deeplabv3 import ASPP, DeepLabV3
from .resunet import ResUNet
from .unet import DoubleConv, UNet

__all__ = [
    "REGISTRY", "PRIMARY_MODELS", "DEFAULT_FEATURES", "META_SUFFIX",
    "DoubleConv", "UNet", "ResUNet", "DeepLabV3", "ASPP", "blocks",
    "list_models", "accepted_options", "model_config", "build_model",
    "build_from_config", "write_meta", "read_meta", "extract_state_dict",
    "load_model", "count_parameters",
]

REGISTRY = {
    "unet": UNet,
    "resunet": ResUNet,
    "deeplabv3": DeepLabV3,
}

PRIMARY_MODELS = ("unet", "resunet", "deeplabv3")

META_SUFFIX = ".meta.json"


def _resolve(name):
    key = str(name).lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(REGISTRY)}")
    return key, REGISTRY[key]


def list_models():
    return list(REGISTRY)


def accepted_options(name):
    """Constructor keyword arguments of an architecture, with their defaults."""
    _, cls = _resolve(name)
    params = inspect.signature(cls.__init__).parameters
    return {
        key: (None if param.default is inspect.Parameter.empty else param.default)
        for key, param in params.items()
        if key != "self"
    }


DEFAULT_FEATURES = {name: list(accepted_options(name)["features"]) for name in REGISTRY}


def _filter_options(key, options):
    """Split options into the ones this architecture accepts and the ones it ignores."""
    accepted = accepted_options(key)
    taken, ignored = {}, []
    for option, value in options.items():
        if value is None:
            continue
        if option in accepted:
            taken[option] = list(value) if isinstance(value, tuple) else value
        else:
            ignored.append(option)
    return taken, ignored


def model_config(name, **options):
    """Normalise constructor options into a fully resolved, JSON-serialisable config.

    Returns ``(config, ignored)``: ``config`` is ``{"model": name, **kwargs}`` with every
    constructor argument present (defaults filled in, ``None`` options meaning "use the
    default"); ``ignored`` lists the options this architecture does not take, for example
    ``norm_mode`` for the plain U-Net.
    """
    key, _ = _resolve(name)
    config = {"model": key}
    for option, default in accepted_options(key).items():
        config[option] = list(default) if isinstance(default, (list, tuple)) else default
    taken, ignored = _filter_options(key, options)
    config.update(taken)
    return config, ignored


def build_from_config(config):
    kwargs = dict(config)
    key, cls = _resolve(kwargs.pop("model"))
    return cls(**kwargs)


def build_model(name, in_ch=1, out_ch=1, **options):
    """Instantiate an architecture by name; unknown options are ignored."""
    config, _ = model_config(name, in_ch=in_ch, out_ch=out_ch, **options)
    return build_from_config(config)


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def write_meta(checkpoint_path, config, **extra):
    """Write ``<checkpoint>.meta.json`` next to a saved state_dict."""
    payload = {"model": config["model"], "config": dict(config), **extra}
    path = checkpoint_path + META_SUFFIX
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def read_meta(checkpoint_path):
    path = checkpoint_path + META_SUFFIX
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def extract_state_dict(checkpoint):
    """Accept a plain state_dict or a training checkpoint that wraps one."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(checkpoint_path, device="cpu", model_name=None, strict=True, **options):
    """Rebuild an architecture and load its weights.

    The architecture is taken from ``model_name`` when given, otherwise from the
    sidecar ``.meta.json``, otherwise it is assumed to be the plain MACS U-Net
    (the format of every checkpoint produced before the sidecar existed).
    Explicit ``options`` (features, norm_mode, ...) override the sidecar.
    Returns ``(model, config)``.
    """
    meta = read_meta(checkpoint_path) or {}
    name = model_name or meta.get("model") or "unet"
    key, _ = _resolve(name)
    config, _ = model_config(key)
    if meta.get("model") == key:
        config.update(meta.get("config", {}))
    overlay, _ = _filter_options(key, options)
    config.update(overlay)
    model = build_from_config(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(state), strict=strict)
    return model, config
