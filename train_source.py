"""Train one single-source binary segmentation model on one domain of one task.

Tasks and domains are soft definitions in ``common/domains.py``: a domain name
points at a dataset directory in the shared ``labeled_datasets`` tree and the
task names the label folder (``membranes`` or ``mitochondria``). The task is
inferred from the domain when it is unambiguous.

Any architecture in the ``archs`` registry is trained with the same command. The
defaults are the MACS source-model training: U-Net, Adam at 1e-4 without weight
decay, Dice + Focal + BCE, 256x256 patches with stride 128, a checkpoint and a
validation pass every 10 epochs (``--val_freq``) and ``<out>_final.pth`` at the end.
MACS loads the best-validation checkpoint ``model_best.pth`` of every source.

    python train_source.py --domain snemi3d --epochs 100                      # membrane task, U-Net
    python train_source.py --task mitochondria --domain lucchi-pp --epochs 100 --model resunet

Checkpoints default to the shared layout (override with ``--out_path``):

    <models_dir>/<task>_domains/<model>/<domain>/model.pth

Outputs for ``--out_path <dir>/model.pth``:

    <dir>/model_epoch<N>.pth           plain state_dict every --save_freq epochs
    <dir>/model_final.pth              plain state_dict of the last epoch
    <dir>/model_final.pth.meta.json    architecture + training configuration (archs.load_model reads it)
    <dir>/model_best.pth[.meta.json]   best validation Dice (every --val_freq epochs, default 10); MACS loads it
    <dir>/model_state.pth              full training state while --resume is on (deleted when done)
    <dir>/model.lock                   held while training runs (host, pid, start time); exit code 3
                                       if another live run holds it, see --force
"""
import argparse
import atexit
import json
import os
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from archs import (PRIMARY_MODELS, build_from_config, count_parameters, list_models,
                   model_config, write_meta)
from common import domains
from common.data import PatchSegDataset
from common.losses import LOSSES, build_loss, dice_coef
from common.optim import OPTIMIZERS, SCHEDULERS, build_optimizer, build_scheduler


# ------------------------------------------------------------------ helpers
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_model_opt(text):
    """``key=value`` for extra constructor options; value may be int, float, bool or a list."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--model_opt expects key=value, got '{text}'")
    key, raw = text.split("=", 1)
    raw = raw.strip()

    def scalar(token):
        low = token.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    value = [scalar(t) for t in raw.split(",")] if "," in raw else scalar(raw)
    return key.strip(), value


def checkpoint_path(out_path, suffix):
    stem = out_path[:-4] if out_path.endswith(".pth") else out_path
    return f"{stem}_{suffix}.pth"


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def boot_id():
    """Identifier of the current boot (Linux); a lock from an earlier boot is always stale."""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError:
        return None


def acquire_lock(lock_path, force=False):
    """Create ``lock_path`` for this run; refuse (return the holder) if a live run holds it.

    A lock from a dead process, or from an earlier boot, of this host is stale and is
    taken over. A lock from another host cannot be checked and is treated as held
    unless ``force`` is set.
    """
    if os.path.isfile(lock_path):
        try:
            with open(lock_path, encoding="utf-8") as stream:
                holder = json.load(stream)
        except (OSError, ValueError):
            holder = {}
        same_host = holder.get("host") == socket.gethostname()
        same_boot = holder.get("boot_id") == boot_id()
        alive = same_host and same_boot and isinstance(holder.get("pid"), int) and pid_alive(holder["pid"])
        if not force and (alive or not same_host):
            return holder
    with open(lock_path, "w", encoding="utf-8") as stream:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "boot_id": boot_id(),
                   "started_at": datetime.now().isoformat(timespec="seconds"),
                   "command": " ".join(sys.argv)}, stream, indent=2)
    return None


def release_lock(lock_path):
    try:
        with open(lock_path, encoding="utf-8") as stream:
            holder = json.load(stream)
        if holder.get("pid") == os.getpid() and holder.get("host") == socket.gethostname():
            os.remove(lock_path)
    except (OSError, ValueError):
        pass


def grads_finite(model):
    """True when every gradient is finite (one device sync, like loss.item())."""
    norms = [torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None]
    return bool(torch.isfinite(torch.stack(norms)).all()) if norms else True


def train_epoch(model, loader, criterion, optimizer, device, description, clip_grad_norm=0.0,
                skip_nonfinite=True):
    """One epoch. With skip_nonfinite (the default) a batch whose loss or gradients are not finite
    is skipped instead of stepping the optimizer, so one bad batch cannot turn every weight into NaN.
    Such a batch is left out of the epoch averages; with no such batch the epoch is unchanged.
    A single-patch batch that BatchNorm cannot train on (DeepLabV3) is skipped the same way.
    Returns (mean loss, mean dice, number of skipped batches)."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    seen = 0
    skipped = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for imgs, masks in progress:
        imgs, masks = imgs.to(device), masks.to(device)
        try:
            preds = model(imgs)
        except ValueError as exc:
            # BatchNorm cannot train on one value per channel: DeepLabV3's image-pooling branch sees
            # 1x1 maps, so a last batch holding a single patch raises. Skip that batch instead of
            # aborting the run; a run that never hits this is unchanged.
            if imgs.size(0) == 1 and "Expected more than 1 value per channel" in str(exc):
                skipped += 1
                progress.set_postfix(loss="skipped", skipped=skipped)
                continue
            raise
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        if skip_nonfinite and not (torch.isfinite(loss).item() and grads_finite(model)):
            # e.g. a batch of all-black padding patches: zero variance in every BatchNorm layer
            # gives NaN gradients in its backward pass. Drop the step, keep the weights.
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            progress.set_postfix(loss="skipped", skipped=skipped)
            continue
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        batch = imgs.size(0)
        seen += batch
        total_loss += loss.item() * batch
        with torch.no_grad():
            total_dice += dice_coef(preds, masks).item() * batch
        progress.set_postfix(loss=f"{loss.item():.4f}")
    n = max(seen, 1)
    return total_loss / n, total_dice / n, skipped


@torch.no_grad()
def val_epoch(model, loader, criterion, device, description):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for imgs, masks in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        batch = imgs.size(0)
        total_loss += criterion(preds, masks).item() * batch
        total_dice += dice_coef(preds, masks).item() * batch
    n = len(loader.dataset)
    return total_loss / n, total_dice / n


# ------------------------------------------------------------------ CLI
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a single-source segmentation model (any architecture in archs/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument('--task', default=None, choices=domains.list_tasks(),
                      help='Segmentation task (label folder); default: inferred from --domain')
    data.add_argument('--domain', required=True,
                      help='Domain name from common/domains.py (python -m common.domains lists them)')
    data.add_argument('--data_root', default=None,
                      help='Flat layout <data_root>/<domain> instead of the registry path')
    data.add_argument('--label_dir', default=None,
                      help='Label folder inside every split (default: the task\'s, membranes / mitochondria)')
    data.add_argument('--patch_size', type=int, default=256)
    data.add_argument('--stride', type=int, default=128)

    arch = parser.add_argument_group("architecture")
    arch.add_argument('--model', default='unet', choices=list_models(),
                      help=f'Architecture; primary choices: {", ".join(PRIMARY_MODELS)}')
    arch.add_argument('--features', type=parse_int_list, default=None,
                      help='Comma-separated channel widths per level (default: the architecture\'s own)')
    arch.add_argument('--norm_mode', default=None, choices=['bn', 'gn', 'in'],
                      help='Normalisation for resunet/deeplabv3 (default bn); the plain unet ignores it')
    arch.add_argument('--act_mode', default=None, choices=['relu', 'leaky_relu', 'elu', 'gelu'],
                      help='Activation for resunet/deeplabv3 (default elu); the plain unet ignores it')
    arch.add_argument('--model_opt', type=parse_model_opt, action='append', default=[], metavar='KEY=VALUE',
                      help='Extra constructor option, repeatable (e.g. atrous_rates=6,12,18)')

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument('--epochs', type=int, default=50)
    optimisation.add_argument('--batch_size', type=int, default=8)
    optimisation.add_argument('--lr', type=float, default=1e-4)
    optimisation.add_argument('--optimizer', default='adam', choices=OPTIMIZERS)
    optimisation.add_argument('--weight_decay', type=float, default=0.0)
    optimisation.add_argument('--momentum', type=float, default=0.9, help='sgd / rmsprop only')
    optimisation.add_argument('--clip_grad_norm', type=float, default=0.0,
                              help='Clip the global gradient norm to this value before each step '
                                   '(0 = off, the default)')
    optimisation.add_argument('--no_skip_nonfinite', dest='skip_nonfinite', action='store_false',
                              help='Step the optimizer even when a batch gives a non-finite loss or gradient '
                                   '(default: skip such a batch; batches with finite gradients are unaffected)')
    optimisation.add_argument('--scheduler', default='none', choices=SCHEDULERS,
                              help='Stepped once per epoch; plateau follows the validation loss')
    optimisation.add_argument('--step_size', type=int, default=20, help='step scheduler')
    optimisation.add_argument('--gamma', type=float, default=0.1, help='step / exponential schedulers')
    optimisation.add_argument('--plateau_patience', type=int, default=10)
    optimisation.add_argument('--plateau_factor', type=float, default=0.5)

    loss = parser.add_argument_group("loss")
    loss.add_argument('--loss', default='combined', choices=LOSSES, help='combined = dice + focal + bce')
    loss.add_argument('--focal_alpha', type=float, default=0.8)
    loss.add_argument('--focal_gamma', type=float, default=2.0)
    loss.add_argument('--tversky_alpha', type=float, default=0.5)
    loss.add_argument('--tversky_beta', type=float, default=0.5)

    output = parser.add_argument_group("checkpoints and logging")
    output.add_argument('--models_dir', default='models',
                        help='Root of the shared layout <models_dir>/<task>_domains/<model>/<domain>/')
    output.add_argument('--out_path', default=None,
                        help='Checkpoint stem; default: <models_dir>/.../<model>/model.pth (see --models_dir)')
    output.add_argument('--force', action='store_true',
                        help='Train even if model.lock is held by another run')
    output.add_argument('--no_lock', action='store_true', help='Do not create model.lock')
    output.add_argument('--save_freq', type=int, default=10, help='Save model_epoch<N>.pth every N epochs (0 = never)')
    output.add_argument('--val_freq', type=int, default=10,
                        help='Validate every N epochs and keep model_best.pth (0 = no validation)')
    output.add_argument('--resume', action='store_true',
                        help='Keep a resumable model_state.pth and continue from it if present')
    output.add_argument('--log_path', default=None, help='Also append the epoch lines to this file')

    misc = parser.add_argument_group("misc")
    misc.add_argument('--num_workers', type=int, default=2, help='DataLoader worker processes')
    misc.add_argument('--seed', type=int, default=None, help='Seed python/numpy/torch (default: unseeded)')
    misc.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


# ------------------------------------------------------------------ main
def main():
    args = build_parser().parse_args()
    device = args.device

    # --- task / domain / paths (soft definitions in common/domains.py)
    flat_layout = bool(args.data_root)
    try:
        args.task, task_spec, domain_spec = domains.resolve(args.domain, args.task, allow_unknown=flat_layout)
    except KeyError as error:
        sys.exit(str(error))
    label_dir = args.label_dir or task_spec.label_dir
    domain_path = domains.domain_path(args.domain, args.task, args.data_root)
    if args.out_path is None:
        args.out_path = os.path.join(
            domains.model_dir(args.models_dir, args.task, args.domain, args.model, allow_unknown=flat_layout),
            "model.pth")
    out_dir = os.path.dirname(args.out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    lock_path = os.path.join(out_dir, "model.lock")
    if not args.no_lock:
        holder = acquire_lock(lock_path, force=args.force)
        if holder is not None:
            print(f"Another run holds {lock_path}: {holder}. Use --force to take it over.", file=sys.stderr)
            sys.exit(3)
        atexit.register(release_lock, lock_path)   # also on errors and Ctrl-C

    def log(message):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        tqdm.write(line)
        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
            with open(args.log_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    if args.seed is not None:
        seed_everything(args.seed)

    # --- model
    options = {"features": args.features, "norm_mode": args.norm_mode, "act_mode": args.act_mode}
    options.update(dict(args.model_opt))
    config, ignored = model_config(args.model, in_ch=1, out_ch=1, **options)
    model = build_from_config(config).to(device)
    if ignored:
        log(f"Note: {args.model} does not take {', '.join(ignored)}; ignored")

    # --- data
    log(f"Task: {args.task} | domain: {domain_spec.name} | labels: {label_dir} | data: {domain_path}")
    if domain_spec.notes:
        log(f"Domain note: {domain_spec.notes}")
    log(f"Checkpoints: {out_dir}")
    log(f"Model: {config['model']} | " + " | ".join(f"{k}={v}" for k, v in config.items() if k != "model"))
    log(f"Parameters: {count_parameters(model):,} ({count_parameters(model, trainable_only=True):,} trainable)")
    log(f"Optimizer: {args.optimizer} lr={args.lr} weight_decay={args.weight_decay} | "
        f"scheduler: {args.scheduler} | loss: {args.loss} | epochs: {args.epochs} | batch: {args.batch_size} | "
        f"patch: {args.patch_size}/{args.stride} | seed: {args.seed} | device: {device}"
        + (f" | clip_grad_norm: {args.clip_grad_norm}" if args.clip_grad_norm > 0 else ""))

    dataset_options = dict(patch_size=args.patch_size, stride=args.stride, label_dir=label_dir,
                           ignore_label=domain_spec.ignore_label)
    train_ds = PatchSegDataset(domain_path, "train", **dataset_options)
    if len(train_ds) == 0:
        sys.exit(f"No training patches in {domain_path} (patch {args.patch_size}, stride {args.stride})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if args.val_freq > 0:
        val_ds = PatchSegDataset(domain_path, "val", **dataset_options)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- optimisation
    criterion = build_loss(args.loss, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                           tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs, step_size=args.step_size,
                                gamma=args.gamma, patience=args.plateau_patience, factor=args.plateau_factor)

    # --- resume
    state_path = checkpoint_path(args.out_path, "state")
    start_epoch = 0
    best_dice = float("-inf")
    if args.resume and os.path.isfile(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        best_dice = state.get("best_dice", best_dice)
        log(f"Resumed from {state_path} at epoch {start_epoch + 1}")

    def save_weights(suffix, **meta):
        path = checkpoint_path(args.out_path, suffix)
        torch.save(model.state_dict(), path)
        write_meta(path, config, task=args.task, domain=domain_spec.name, label_dir=label_dir,
                   data_path=os.path.abspath(domain_path), data_root=args.data_root, training=vars(args),
                   num_parameters=count_parameters(model), train_patches=len(train_ds),
                   torch_version=torch.__version__, saved_at=datetime.now().isoformat(timespec="seconds"),
                   **meta)
        return path

    def save_state(epoch):
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_dice": best_dice,
            "config": config,
        }, state_path)

    # --- training loop
    started = time.perf_counter()
    train_loss = train_dice = float("nan")
    for epoch in range(start_epoch, args.epochs):
        tag = f"Epoch {epoch + 1}/{args.epochs}"
        train_loss, train_dice, skipped = train_epoch(model, train_loader, criterion, optimizer, device, tag,
                                                          clip_grad_norm=args.clip_grad_norm,
                                                          skip_nonfinite=args.skip_nonfinite)
        message = f"{tag} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}"
        if skipped:
            message += f" | skipped {skipped} non-finite batch(es)"

        val_loss = None
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            val_loss, val_dice = val_epoch(model, val_loader, criterion, device, f"{tag} (Val)")
            message += f" | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
            if val_dice > best_dice:
                best_dice = val_dice
                path = save_weights("best", epoch=epoch + 1, val_loss=val_loss, val_dice=val_dice)
                message += f" | new best -> {path}"

        if scheduler is not None:
            if args.scheduler == "plateau":
                if val_loss is not None:
                    scheduler.step(val_loss)
            else:
                scheduler.step()
            message += f" | lr: {optimizer.param_groups[0]['lr']:.2e}"
        log(message)

        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            path = checkpoint_path(args.out_path, f"epoch{epoch + 1}")
            torch.save(model.state_dict(), path)
            log(f"Model checkpoint saved at {path}")
            if args.resume:
                save_state(epoch + 1)

    final_path = save_weights(
        "final", epochs_trained=args.epochs, final_train_loss=train_loss, final_train_dice=train_dice,
        best_val_dice=(None if best_dice == float("-inf") else best_dice),
        elapsed_seconds=round(time.perf_counter() - started, 1))
    if os.path.isfile(state_path):
        os.remove(state_path)
    log(f"Final model saved at {final_path}")


if __name__ == '__main__':
    main()
