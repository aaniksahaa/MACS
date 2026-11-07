"""One MACS run on one target: MK-MMD source weights, calibrated ensemble distillation into the student, Laplace
active learning with label-driven weight correction, consolidation, test evaluation."""
import csv
import dataclasses
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset

from .calibration import collect_pixels, fit_temperature
from .config import Config
from .evaluate import evaluate_model
from .features import FeatureStore, extract
from .student import STUDENT_FEATURES, active_learning, adapt_bn, blank_patches, build_student, kd_train
from .teachers import TeacherPool
from .utils import (SHORT, Logger, Resources, canonical, environment, file_tag, make_dataset, sample_indices,
                    save_json, seed_everything, task_domains, weight_stats)
from .weights import common_embedding, mixture_weights

DATA_ACCESS = ("MMD weights and distillation use the unlabeled target train pool only; teacher temperatures "
               "are fitted on each teacher's own source validation set; target validation labels are used "
               "only for reported diagnostics and per-round monitoring; test labels are used once, for the "
               "final evaluation.")


class Datasets:
    """Memoised PatchSegDataset per (domain, split)."""

    def __init__(self):
        self.cache = {}

    def get(self, domain, split):
        key = (canonical(domain), split)
        if key not in self.cache:
            self.cache[key] = make_dataset(domain, split)
        return self.cache[key]


def parse_sources(cfg, target):
    """(domains, architectures) of the source pool: --sources 'domain[:arch],...' or every other domain."""
    specs = [s.strip() for s in cfg.sources.split(",") if s.strip()] or [d for d in task_domains() if d != target]
    archs = [s.split(":")[1] if ":" in s else cfg.arch for s in specs]
    return [canonical(s.split(":")[0]) for s in specs], archs


def features_of(store, datasets, cfg, data_domain, split, encoder_index, pool, device):
    ds = datasets.get(data_domain, split)
    idx = sample_indices(len(ds), cfg.feature_n, cfg.feature_seed, canonical(data_domain), split, cfg.feature_n)
    return store.get(canonical(data_domain), split, cfg.feature_n, cfg.feature_seed, pool.sources[encoder_index],
                     pool.archs[encoder_index], pool.paths[encoder_index],
                     lambda: extract(pool.models[encoder_index], ds, idx, device, cfg.eval_batch_size, cfg.num_workers))


def compute_weights(cfg, target, pool, store, datasets, device, log):
    """MK-MMD mixture weights of the sources for one target: every domain is encoded by every teacher."""
    t0 = time.time()
    m = len(pool)
    src_common, hits = [], 0
    tgt_blocks = []
    for e in range(m):
        arr, hit = features_of(store, datasets, cfg, target, "train", e, pool, device)
        hits += hit
        tgt_blocks.append(arr)
    for i in range(m):
        blocks = []
        for e in range(m):
            arr, hit = features_of(store, datasets, cfg, pool.sources[i], cfg.source_split, e, pool, device)
            hits += hit
            blocks.append(arr)
        src_common.append(common_embedding(blocks))
    tgt_common = common_embedding(tgt_blocks)
    log("FEATURES", f"{m * (m + 1)} domain x encoder feature sets ({hits} cached) | n={cfg.feature_n}/domain | {time.time() - t0:.0f}s")
    scales = [float(s) for s in cfg.kernel_scales.split(",") if s.strip()]
    mix, mix_info = mixture_weights(src_common, tgt_common, scales, cfg.lam, seed=cfg.feature_seed)
    log("WEIGHTS", "mixture: " + ", ".join(f"{SHORT[s]}={w:.3f}" for s, w in zip(pool.sources, mix))
        + f" | eff.sources={mix_info['effective_sources']:.2f} | MMD2 mix={mix_info['mmd2_mixture']:.4f} uniform={mix_info['mmd2_uniform']:.4f}")
    return {"sources": list(pool.sources), "mixture": mix.tolist(), "mixture_info": mix_info, "lam": cfg.lam,
            "seconds": round(time.time() - t0, 1)}


def teacher_temperatures(cfg, pool, datasets, device, log):
    """Temperature of every teacher, fitted on its own source validation split (cached per checkpoint)."""
    temps = np.ones(len(pool))
    tdir = os.path.join(cfg.cache_dir, "temperatures")
    os.makedirs(tdir, exist_ok=True)
    for i, domain in enumerate(pool.sources):
        path = os.path.join(tdir, f"{pool.archs[i]}-{domain}-{file_tag(pool.paths[i])}.json")
        if os.path.isfile(path):
            temps[i] = json.load(open(path))["temperature"]
            continue
        ds = datasets.get(domain, "val")
        idx = sample_indices(len(ds), 256, 0, "calib", domain)
        logits, labels = collect_pixels(pool.models[i], ds, idx, device, num_workers=cfg.num_workers)
        temps[i] = fit_temperature(logits, labels)
        save_json(path, {"temperature": temps[i], "domain": domain, "pixels": int(len(labels))})
    log("CALIB", "temperatures: " + ", ".join(f"{SHORT[s]}={t:.2f}" for s, t in zip(pool.sources, temps)))
    return temps


def teacher_diagnostics(cfg, pool, datasets, target, weight_sets, log):
    """Teacher reliability on the target validation split (labels used for reporting only)."""
    ds = datasets.get(target, "val")
    idx = sample_indices(len(ds), cfg.val_cap, 0, "val", target)
    per_teacher, per_set = pool.target_diagnostics(ds, idx, weight_sets, cfg.eval_batch_size, cfg.num_workers)
    log("TEACHERS", "target-val Dice: " + ", ".join(f"{SHORT[s]}={d:.3f}" for s, d in zip(pool.sources, per_teacher)))
    log("TEACHERS", "ensemble target-val Dice: " + ", ".join(f"{k}={v:.3f}" for k, v in per_set.items()))
    return {"teacher_val_dice": per_teacher, "ensemble_val_dice": per_set, "val_patches": int(len(idx))}


def run_dir(cfg):
    name = f"b{cfg.budget:g}_s{cfg.seed}" + (f"_{cfg.run_tag}" if cfg.run_tag else "")
    return os.path.join(cfg.output_dir, cfg.task, canonical(cfg.target), name)


def run(cfg: Config):
    target = canonical(cfg.target)
    rdir = run_dir(cfg)
    os.makedirs(rdir, exist_ok=True)
    log = Logger(os.path.join(rdir, "log.txt"))
    device = torch.device(cfg.device)
    seed_everything(cfg.seed)
    t_start = time.time()
    log("START", f"target={target} budget={cfg.budget}% seed={cfg.seed} | " + json.dumps(
        {k: v for k, v in dataclasses.asdict(cfg).items() if k not in ("models_dir", "cache_dir", "output_dir")}))
    log("DATA", DATA_ACCESS)
    datasets = Datasets()
    train_ds, val_ds, test_ds = datasets.get(target, "train"), datasets.get(target, "val"), datasets.get(target, "test")
    log("DATA", f"{target}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} patches")
    result = {"task": cfg.task, "target": target, "budget": cfg.budget, "seed": cfg.seed, "run_tag": cfg.run_tag,
              "config": dataclasses.asdict(cfg), "data_access": DATA_ACCESS, "environment": environment()}
    res = Resources(device)

    # 1. teachers, calibration, MK-MMD weights
    res.start("setup")
    sources, archs = parse_sources(cfg, target)
    pool = TeacherPool(cfg.models_dir, sources, archs, device, cfg.teacher_ckpt)
    pool.temperatures = teacher_temperatures(cfg, pool, datasets, device, log)
    res.stop()
    res.start("features_and_weights")
    w = compute_weights(cfg, target, pool, FeatureStore(cfg.cache_dir), datasets, device, log)
    m = len(sources)
    diag = teacher_diagnostics(cfg, pool, datasets, target, {"mixture": w["mixture"], "uniform": np.full(m, 1.0 / m)}, log)
    alphas = np.asarray(w["mixture"])
    pool.alphas = alphas
    top = int(np.argmax(alphas))
    result.update({"sources": sources, "source_archs": archs, "alphas": alphas.tolist(), "weights": w,
                   "teacher_diagnostics": diag, "temperatures": pool.temperatures.tolist(), "top_source": sources[top],
                   "weight_stats": weight_stats(alphas)})

    # 2. student initialised from the top-weighted teacher (a student-shaped U-Net), BatchNorm adapted to the target
    def student_shaped(i):
        return (type(pool.models[i]).__name__ == "UNet"
                and list(pool.configs[i].get("features", STUDENT_FEATURES)) == STUDENT_FEATURES)
    init_from = top if student_shaped(top) else None
    if init_from is None and cfg.init_compatible:
        init_from = next((int(i) for i in np.argsort(-alphas) if student_shaped(int(i))), None)
    init_state = None
    if init_from is not None:
        init_state = {k: v.clone() for k, v in pool.models[init_from].state_dict().items()}
        result["init_teacher"] = sources[init_from]
        note = "" if init_from == top else f"; top teacher {SHORT[sources[top]]} is not student-shaped"
        log("INIT", f"student initialised from teacher {SHORT[sources[init_from]]} (weight {alphas[init_from]:.3f}{note})")
    else:
        log("INIT", f"top teacher {SHORT[sources[top]]} is not a student-shaped U-Net; random init")
    student = build_student(device, init_state)
    result["student_init"] = "top-source" if init_state is not None else "random"
    if init_state is not None:
        adapt_bn(student, train_ds, np.arange(len(train_ds)), cfg, device, log, "init")
    blank = blank_patches(train_ds, cfg, target, log)
    result["blank_fraction"] = float(blank.mean())

    val_idx = sample_indices(len(val_ds), cfg.val_cap, 0, "val", target)

    def evaluate_round(model):
        r = evaluate_model(model, Subset(val_ds, [int(i) for i in val_idx]) if len(val_idx) < len(val_ds) else val_ds,
                           device, cfg.eval_batch_size, cfg.num_workers, image_level=False, amp=cfg.amp)
        return {"val_dice": r["patch_dice"], "val_vi": r["patch_vi"], "val_ece": r["cal_ece"]}

    # 3. distillation of the calibrated weighted ensemble on unlabeled target patches
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.kd_lr)
    res.stop()
    res.start("distillation")
    t0 = time.time()
    log("KD", "distilling the weighted ensemble into the student")
    optimizer = kd_train(student, pool, train_ds, np.arange(len(train_ds)), cfg, device, log, optimizer)
    result["kd_seconds"] = round(time.time() - t0, 1)
    res.stop({"kd_epochs": cfg.kd_epochs})
    r0 = evaluate_round(student)
    log("ROUND0", f"before any target label: val_dice={r0['val_dice']:.4f} val_vi={r0['val_vi']:.4f}")
    result["round0_val"] = r0

    # 4. active learning with weight correction and pseudo-labels, then consolidation
    for g in optimizer.param_groups:
        g["lr"] = cfg.ft_lr
    t0 = time.time()
    res.start("active_learning")
    rounds, quality = active_learning(student, train_ds, cfg, device, log, optimizer, pool, evaluate_round, blank)
    res.stop({"rounds": cfg.rounds})
    result.update({"rounds": rounds, "uncertainty_quality": quality, "al_seconds": round(time.time() - t0, 1),
                   "alphas_final": np.asarray(pool.alphas).tolist(), "weight_stats_final": weight_stats(pool.alphas),
                   "top_source_final": pool.sources[int(np.argmax(pool.alphas))],
                   "ensemble_choice_final": rounds[-1].get("ensemble_choice", "")})
    log("WEIGHTS", "final: " + ", ".join(f"{SHORT[s]}={a:.3f}" for s, a in zip(pool.sources, pool.alphas)))
    with open(os.path.join(rdir, "rounds.csv"), "w", newline="") as fh:
        fields = []
        for rec in rounds:
            fields += [k for k in rec if k not in fields]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rounds)

    # 5. test evaluation (with and without test-time augmentation) and checkpoint
    t0 = time.time()
    res.start("inference")
    test = evaluate_model(student, test_ds, device, cfg.eval_batch_size, cfg.num_workers, cfg.image_level, amp=cfg.amp,
                          tta=cfg.tta)
    if cfg.tta:
        notta = evaluate_model(student, test_ds, device, cfg.eval_batch_size, cfg.num_workers, False, amp=cfg.amp)
        result["test_notta"] = notta
        log("TEST", f"without TTA: patch_dice={notta['patch_dice']:.4f} patch_vi={notta['patch_vi']:.4f}")
    inf = res.stop({"patches": len(test_ds), "images": test.get("n_images", 0)})
    inf["patches_per_second"] = round(len(test_ds) / max(inf["seconds"], 1e-6), 1)
    result["test"] = test
    result["eval_seconds"] = round(time.time() - t0, 1)
    result["total_seconds"] = round(time.time() - t_start, 1)
    result["resources"] = res.summary()
    result["student_parameters"] = int(sum(p.numel() for p in student.parameters()))
    log("RESOURCES", json.dumps(result["resources"]))
    log("TEST", " ".join(f"{k}={v:.4f}" for k, v in test.items() if isinstance(v, float)))
    ckpt = os.path.join(rdir, "model.pth")
    torch.save(student.state_dict(), ckpt)
    result["checkpoint"] = ckpt
    save_json(os.path.join(rdir, "result.json"), result)
    append_result(cfg, result)
    log("DONE", f"{target} b{cfg.budget:g} s{cfg.seed} | test patch Dice={test['patch_dice']:.4f} VI={test['patch_vi']:.4f}"
        + (f" | image Dice={test['image_dice']:.4f} VI={test['image_vi']:.4f}" if "image_dice" in test else "")
        + f" | {result['total_seconds']:.0f}s")
    log.close()
    return result


RESULT_FIELDS = ["task", "target", "budget", "seed", "run_tag", "sources", "top_source", "top_source_final",
                 "patch_dice", "patch_iou", "patch_recall", "patch_vi", "image_dice", "image_iou", "image_recall",
                 "image_vi", "patch_dice_notta", "total_seconds", "peak_gpu_alloc_mb", "checkpoint"]


def append_result(cfg, result):
    """One row per finished run in <output_dir>/results.csv."""
    path = os.path.join(cfg.output_dir, "results.csv")
    new = not os.path.isfile(path)
    t = result["test"]
    row = {"task": cfg.task, "target": result["target"], "budget": cfg.budget, "seed": cfg.seed, "run_tag": cfg.run_tag,
           "sources": ";".join(f"{s}:{a}" for s, a in zip(result["sources"], result["source_archs"])),
           "top_source": result["top_source"], "top_source_final": result["top_source_final"],
           **{k: t.get(k, "") for k in ("patch_dice", "patch_iou", "patch_recall", "patch_vi", "image_dice", "image_iou",
                                        "image_recall", "image_vi")},
           "patch_dice_notta": result.get("test_notta", {}).get("patch_dice", ""),
           "total_seconds": result["total_seconds"], "peak_gpu_alloc_mb": result["resources"].get("peak_gpu_alloc_mb", ""),
           "checkpoint": result["checkpoint"]}
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            wr.writeheader()
        wr.writerow(row)
