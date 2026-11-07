"""Student: knowledge distillation from the calibrated weighted ensemble, Laplace acquisition, label-driven
correction of the source weights, and fine-tuning with pseudo-label (consistency) regularisation."""
import dataclasses
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from archs import UNet
from common.losses import CombinedLoss

from .utils import SHORT, make_loader, sample_indices
from .evaluate import patch_dice

STUDENT_FEATURES = [64, 128, 256, 512]


def build_student(device, init_state=None):
    model = UNet(in_ch=1, out_ch=1, features=STUDENT_FEATURES).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    return model


def penultimate(model, x):
    """(features before final_conv [B, D, H, W], logits [B, 1, H, W]) of the student U-Net."""
    feats = model.get_features(x)
    return feats, model.final_conv(feats)


def augment_batch(imgs, targets, gen):
    """Random flips and 90-degree rotations, identical for image and target (dihedral group)."""
    out_i, out_t = [], []
    for i in range(imgs.shape[0]):
        k = int(torch.randint(0, 8, (1,), generator=gen))
        x, y = imgs[i], targets[i]
        if k & 1:
            x, y = x.flip(-1), y.flip(-1)
        if k & 2:
            x, y = x.flip(-2), y.flip(-2)
        if k & 4:
            x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        out_i.append(x)
        out_t.append(y)
    return torch.stack(out_i), torch.stack(out_t)


def _autocast(device, amp):
    return torch.autocast(device_type="cuda", enabled=amp and str(device).startswith("cuda"))


def blank_patches(dataset, cfg, name, log):
    """Patches without tissue (image std < cfg.blank_std: reflect padding, empty tiles). Their label is known
    without an annotator, so they never cost budget. Cached per domain."""
    path = os.path.join(cfg.cache_dir, "blank", f"{name}_train_{dataset.patch_size}_{len(dataset)}.npy")
    if os.path.isfile(path):
        return np.load(path)
    t0 = time.time()
    stds = []
    for imgs, _ in make_loader(dataset, None, 64, cfg.num_workers):
        stds.extend(imgs.flatten(1).std(1).tolist())
    blank = np.asarray(stds) < cfg.blank_std
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, blank)
    log("POOL", f"{int(blank.sum())}/{len(blank)} blank patches found in {time.time() - t0:.0f}s")
    return blank


class FreeBlankSet(torch.utils.data.Dataset):
    """Labeled patches plus blank patches whose label (all background) costs no annotation. A blank patch
    is served with an all-zero mask, whatever the file says: no annotator ever looked at it."""

    def __init__(self, base, labeled, blank):
        self.base = base
        self.items = [(int(i), False) for i in labeled] + [(int(i), True) for i in blank]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i, is_blank = self.items[j]
        img, mask = self.base[i]
        return img, (torch.zeros_like(mask) if is_blank else mask)


@torch.no_grad()
def adapt_bn(model, dataset, indices, cfg, device, log, tag):
    """AdaBN (Li et al. 2016): re-estimate every BatchNorm running mean / variance on unlabeled target patches.
    No labels, no gradients; applied to a student that was just loaded from a teacher."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns or cfg.adabn_cap <= 0:
        return
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # cumulative average over the pass
    model.train()
    sub = np.asarray(indices)[sample_indices(len(indices), cfg.adabn_cap, cfg.seed, "adabn", tag)]
    for imgs, _ in make_loader(dataset, sub, cfg.eval_batch_size, cfg.num_workers):
        with _autocast(device, cfg.amp):
            model(imgs.to(device, non_blocking=True))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.eval()
    log("ADABN", f"{tag}: BatchNorm statistics re-estimated on {len(sub)} unlabeled target patches")


class SoftLabelSet(torch.utils.data.Dataset):
    """Target patches paired with precomputed soft labels (uint8 maps, 1/255 resolution). `ens_soft` keeps the
    ensemble's labels after `soft` is switched to the student's, so the pseudo-label source can switch back."""

    def __init__(self, base, indices, soft):
        self.base, self.indices, self.soft = base, [int(i) for i in indices], soft
        self.ens_soft = soft

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        img, _ = self.base[self.indices[j]]
        return img, self.soft[j].float().div_(255.0).unsqueeze(0)

    def remove(self, indices):
        drop = set(int(i) for i in indices)
        keep = [j for j, i in enumerate(self.indices) if i not in drop]
        if len(keep) == len(self.indices):
            return
        self.indices = [self.indices[j] for j in keep]
        self.soft = self.soft[keep]
        self.ens_soft = self.ens_soft[keep]


@torch.no_grad()
def ensemble_soft_labels(pool, dataset, indices, cfg, device):
    """One pass of every teacher over the patches; the weighted ensemble is stored as uint8 on the CPU."""
    for m in pool.models:
        m.eval()
    out = torch.empty((len(indices), dataset.patch_size, dataset.patch_size), dtype=torch.uint8)
    pos = 0
    for imgs, _ in make_loader(dataset, indices, cfg.eval_batch_size, cfg.num_workers):
        with _autocast(device, cfg.amp):
            soft = pool.ensemble(imgs.to(device, non_blocking=True)).float()
        n = soft.shape[0]
        out[pos:pos + n] = (soft[:, 0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        pos += n
    return out


def kd_train(student, pool, dataset, pool_indices, cfg, device, log, optimizer=None):
    """Distil the weighted teacher ensemble into the student on a fixed random subset of the unlabeled
    target pool (at most cfg.kd_cap patches). Soft labels are computed once (one pass per teacher), then every
    epoch costs only the student pass."""
    if optimizer is None:
        optimizer = torch.optim.Adam(student.parameters(), lr=cfg.kd_lr)
    pool_indices = np.asarray(pool_indices)
    subset = pool_indices[sample_indices(len(pool_indices), cfg.kd_cap, cfg.seed, "kd")]
    t0 = time.time()
    soft = ensemble_soft_labels(pool, dataset, subset, cfg, device)
    log("KD", f"soft labels for {len(subset)} patches from {len(pool)} teachers in {time.time() - t0:.0f}s")
    kd_set = SoftLabelSet(dataset, subset, soft)
    for epoch in range(cfg.kd_epochs):
        loader = torch.utils.data.DataLoader(kd_set, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.num_workers, pin_memory=True)
        student.train()
        total, n = 0.0, 0
        t0 = time.time()
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        for imgs, soft_b in loader:
            if cfg.augment:
                imgs, soft_b = augment_batch(imgs, soft_b, gen)
            imgs = imgs.to(device, non_blocking=True)
            soft_b = soft_b.to(device, non_blocking=True).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy_with_logits(student(imgs), soft_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        log("KD", f"epoch {epoch + 1}/{cfg.kd_epochs} | patches={n} | loss={total / max(n, 1):.4f} | {time.time() - t0:.0f}s")
    return optimizer


# ------------------------------------------------------------------ uncertainty / acquisition
@torch.no_grad()
def laplace_hessian(student, dataset, indices, cfg, device):
    """Diagonal Hessian (GGN) of the last layer: sum over pixels of p(1-p) z^2."""
    student.eval()
    acc = None
    for imgs, _ in make_loader(dataset, indices, cfg.eval_batch_size, cfg.num_workers):
        with _autocast(device, cfg.amp):
            feats, logits = penultimate(student, imgs.to(device, non_blocking=True))
        feats, logits = feats.float(), logits.float()
        p = torch.sigmoid(logits)
        pp = (p * (1 - p))
        term = (pp * feats.pow(2)).sum(dim=(0, 2, 3))
        acc = term if acc is None else acc + term
    if acc is None:                                   # no unlabeled patches left (budget >= pool)
        return torch.ones(STUDENT_FEATURES[0], device=device)
    return acc + 1e-6


@torch.no_grad()
def laplace_scores(student, dataset, indices, cfg, device, hess, with_error=False):
    """Per-patch Laplace score (higher = more uncertain): predictive entropy + half the last-layer Laplace variance
    sum_d z_d^2 / H_dd, mean over pixels. Also the predicted foreground fraction of every candidate and, with
    with_error, the patch error 1 - Dice (diagnostic, uses the labels of the candidates)."""
    student.eval()
    indices = np.asarray(indices)
    w_var = (1.0 / hess).view(1, -1, 1, 1)
    scores, errors, prevalence = [], [], []
    for imgs, masks in make_loader(dataset, indices, cfg.eval_batch_size, cfg.num_workers):
        imgs = imgs.to(device, non_blocking=True)
        with _autocast(device, cfg.amp):
            feats, logits = penultimate(student, imgs)
        feats, logits = feats.float(), logits.float()
        p = torch.sigmoid(logits)
        ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
        var = (feats.pow(2) * w_var).sum(dim=1, keepdim=True)
        u = 0.5 * var + ent
        scores.extend(u.mean(dim=(1, 2, 3)).tolist())
        prevalence.extend((p > 0.5).float().mean(dim=(1, 2, 3)).tolist())
        if with_error:
            errors.extend((1 - patch_dice(p, masks.to(device))).tolist())
    return np.asarray(scores), (np.asarray(errors) if with_error else None), np.asarray(prevalence)


def masks_fg(dataset, indices):
    """Foreground pixels in the masks of the given patches (for the pseudo-source judgement gate)."""
    total = 0
    for _, m in make_loader(dataset, indices, 64, 2):
        total += int((m > 0.5).sum())
    return total


def strat_select(scores, prevalence, k, bins):
    """Stratified uncertainty sampling: candidates are binned by predicted foreground fraction (quantiles), every
    bin gets its share of the batch, and the most uncertain patches of each bin are taken. Uncertainty grows with
    the amount of boundary in a patch, so plain top-k spends the budget on foreground-dense near-duplicates; the
    strata keep the batch representative of the pool. Non-finite scores are never chosen."""
    ok = np.isfinite(scores)
    edges = np.quantile(prevalence[ok], np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, prevalence, side="right") - 1, 0, bins - 1)
    share = [k // bins + (1 if i < k % bins else 0) for i in range(bins)]
    out = []
    for i in range(bins):
        idx = np.flatnonzero((b == i) & ok)
        out += list(idx[np.argsort(-scores[idx])[:share[i]]])
    taken = set(out)
    rest = [i for i in np.argsort(-scores) if ok[i] and i not in taken]      # bins short of candidates: top up by score
    return np.asarray(out + rest[:k - len(out)], dtype=int)


def uncertainty_quality(scores, errors):
    if errors is None or len(scores) < 10:
        return {}
    rho = spearmanr(scores, errors).correlation
    top = errors >= np.quantile(errors, 0.8)
    auroc = roc_auc_score(top, scores) if 0 < top.sum() < len(top) else float("nan")
    return {"spearman_uncertainty_error": float(rho), "auroc_top20_error": float(auroc)}


# ------------------------------------------------------------------ fine-tuning
@torch.no_grad()
def student_soft_labels(student, dataset, indices, cfg, device):
    student.eval()
    out = torch.empty((len(indices), dataset.patch_size, dataset.patch_size), dtype=torch.uint8)
    pos = 0
    for imgs, _ in make_loader(dataset, indices, cfg.eval_batch_size, cfg.num_workers):
        with _autocast(device, cfg.amp):
            p = torch.sigmoid(student(imgs.to(device, non_blocking=True)).float())
        n = p.shape[0]
        out[pos:pos + n] = (p[:, 0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        pos += n
    return out


@torch.no_grad()
def student_dice(student, dataset, indices, cfg, device):
    """Dice of all labeled pixels taken together (pooled over the patches)."""
    student.eval()
    acc = [0.0, 0.0, 0.0]
    for imgs, masks in make_loader(dataset, indices, cfg.eval_batch_size, cfg.num_workers):
        masks = masks.to(device)
        with _autocast(device, cfg.amp):
            p = torch.sigmoid(student(imgs.to(device, non_blocking=True)).float())
        pred = (p > 0.5).float()
        acc[0] += float((pred * masks).sum()); acc[1] += float(pred.sum()); acc[2] += float(masks.sum())
    return (2 * acc[0] + 1.0) / (acc[1] + acc[2] + 1.0)


def build_consistency_set(pool, dataset, unlabeled, cfg, device, log):
    """Ensemble pseudo-labels on a fixed random subset of the unlabeled pool (computed once per weight change)."""
    sub = np.asarray(unlabeled)[sample_indices(len(unlabeled), cfg.consistency_cap, cfg.seed, "consistency")]
    t0 = time.time()
    soft = ensemble_soft_labels(pool, dataset, sub, cfg, device)
    log("CONS", f"pseudo-labels for {len(sub)} unlabeled patches from the current ensemble in {time.time() - t0:.0f}s")
    return SoftLabelSet(dataset, sub, soft)


def finetune(student, dataset, labeled, cfg, device, log, optimizer, round_label="", consistency_set=None,
             blank_idx=None):
    """Supervised loss on the labeled patches (plus as many free blank negatives) and, with a consistency set,
    cfg.consistency x soft-target BCE against its pseudo-labels."""
    criterion = CombinedLoss() if cfg.ft_loss == "combined" else torch.nn.BCEWithLogitsLoss()
    labeled = np.asarray(labeled)
    if consistency_set is not None and 0 < len(consistency_set) < cfg.min_consistency:
        # a near-exhausted pool (budgets >= the non-blank pool) leaves a handful of pseudo-labelled patches that the
        # consistency loader would cycle hundreds of times per epoch: supervised fine-tuning only in that case
        log("FT", f"{round_label}: consistency set has {len(consistency_set)} patches (< {cfg.min_consistency}); supervised only")
        consistency_set = None
    for epoch in range(cfg.ft_epochs):
        subset = labeled[sample_indices(len(labeled), cfg.ft_cap, cfg.seed, "ft", round_label, epoch)]
        if blank_idx is not None and len(blank_idx) > 0:
            # as many free blank negatives as labeled patches, a fresh draw every epoch
            free = np.asarray(blank_idx)[sample_indices(len(blank_idx), len(subset), cfg.seed, "free", round_label, epoch)]
            loader = torch.utils.data.DataLoader(FreeBlankSet(dataset, subset, free), batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers, pin_memory=True)
        else:
            loader = make_loader(dataset, subset, cfg.batch_size, cfg.num_workers, shuffle=True)
        unl_iter = None
        if cfg.consistency > 0 and consistency_set is not None and len(consistency_set) > 0:
            unl_iter = iter(torch.utils.data.DataLoader(consistency_set, batch_size=cfg.batch_size, shuffle=True,
                                                        num_workers=cfg.num_workers, pin_memory=True))
        student.train()
        total, n = 0.0, 0
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch + len(labeled))
        for imgs, masks in loader:
            if cfg.augment:
                imgs, masks = augment_batch(imgs, masks, gen)
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(student(imgs), masks)
            loss.backward()                       # the supervised graph is freed before the consistency batch
            if unl_iter is not None:
                try:
                    u_imgs, soft = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(torch.utils.data.DataLoader(consistency_set, batch_size=cfg.batch_size, shuffle=True,
                                                                num_workers=cfg.num_workers, pin_memory=True))
                    u_imgs, soft = next(unl_iter)
                if cfg.augment:
                    u_imgs, soft = augment_batch(u_imgs, soft, gen)
                u_imgs = u_imgs.to(device, non_blocking=True)
                soft = soft.to(device, non_blocking=True).clamp(1e-4, 1 - 1e-4)
                cons = cfg.consistency * F.binary_cross_entropy_with_logits(student(u_imgs), soft)
                cons.backward()
                loss = loss + cons.detach()
            optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        log("FT", f"{round_label} epoch {epoch + 1}/{cfg.ft_epochs} | patches={n} | loss={total / max(n, 1):.4f}")


def evidence_set(train_ds, idx, blank_idx, cfg, tag):
    """Labeled patches plus as many free blank tiles (all-background masks): the labeled evidence must also
    charge a teacher for foreground hallucinated on empty tiles, as the patch-level test metric does."""
    if blank_idx is None or len(blank_idx) == 0:
        return train_ds, idx
    free = np.asarray(blank_idx)[sample_indices(len(blank_idx), len(idx), cfg.seed, "evidence", tag)]
    return FreeBlankSet(train_ds, idx, free), None


def update_weights(student, pool, train_ds, labeled, cfg, device, log, r, optimizer, blank_idx=None,
                   pool_idx=None, student_ref=False):
    """Correct the source weights with the labels acquired so far.

    alpha_i <- alpha_prior_i * exp(beta * Dice_i), normalised: the MK-MMD weights are the prior, the teachers' Dice
    on the labeled pixels (pooled over the labeled patches plus as many free blank tiles) is the evidence. Nothing
    changes until cfg.min_evidence_fg foreground pixels have been labeled. Once the student labels the pool itself
    (student_ref), every Dice is blended (cfg.pool_evidence) with the teacher's agreement with the student on
    unlabeled patches. The labeled evidence then chooses among candidate ensembles (the corrected mixture, its top-k
    restrictions and uniform). The first round that passes the gate is the correction round: the caller distils
    the corrected ensemble into the student (logged: the student's and the best teacher's Dice on these pixels)."""
    idx = labeled[sample_indices(len(labeled), cfg.update_cap, cfg.seed, "update", r)]
    ev_ds, ev_idx = evidence_set(train_ds, idx, blank_idx, cfg, r)
    dice = np.asarray(pool.teacher_dice(ev_ds, ev_idx, cfg.eval_batch_size, cfg.num_workers, pooled=True))
    ev = dict(pool.evidence)
    ref_ds = None
    if cfg.pool_evidence > 0 and pool_idx is not None and student_ref and len(pool_idx) > 0:
        sub = np.asarray(pool_idx)[sample_indices(len(pool_idx), cfg.pool_evidence_n, cfg.seed, "poolev", r)]
        hard = (student_soft_labels(student, train_ds, sub, cfg, device) >= 128).to(torch.uint8) * 255
        ref_ds = SoftLabelSet(train_ds, sub, hard)
        dice_pool = np.asarray(pool.teacher_dice(ref_ds, np.arange(len(sub)), cfg.eval_batch_size, cfg.num_workers, pooled=True))
        pool.evidence = ev                     # keep the labeled evidence record (fg pixels, gate)
        log("WEIGHTS", f"round {r}: teacher agreement with the student on {len(sub)} unlabeled patches: " + ", ".join(
            f"{SHORT[s_]}={d:.3f}" for s_, d in zip(pool.sources, dice_pool)) + f" | blended with labeled Dice at {cfg.pool_evidence:.2f}")
        rec_pool = dice_pool.tolist()
        dice = (1 - cfg.pool_evidence) * dice + cfg.pool_evidence * dice_pool
    else:
        rec_pool = None
    if not hasattr(pool, "alphas_prior"):
        pool.alphas_prior = np.asarray(pool.alphas, dtype=np.float64).copy()
        pool.corrected = False
    rec = {"teacher_dice_labeled": dice.tolist(), "teacher_dice_pool": rec_pool, "evidence_fg_pixels": ev["fg_pixels"],
           "evidence_nonempty_patches": ev["nonempty_patches"], "first_correction": False}
    if ev["fg_pixels"] < cfg.min_evidence_fg:
        log("WEIGHTS", f"round {r}: {ev['fg_pixels']} labeled foreground pixels in {len(idx)} patches "
                       f"(< {cfg.min_evidence_fg}): weights stay at the prior")
        rec.update({"weights": np.asarray(pool.alphas).tolist(), "ensemble_choice": "prior", "candidate_dice": {}})
        return rec
    first = not pool.corrected
    pool.corrected = True
    rec["first_correction"] = first
    prior = np.maximum(pool.alphas_prior, 1e-6)
    new = prior * np.exp(cfg.reliability_beta * (dice - dice.max()))
    new /= new.sum()
    # the labeled evidence chooses among candidate combinations; ties within 0.005 favour more sources
    order = np.argsort(-dice)
    cands = {"mixture": new, "uniform": np.full(len(pool), 1.0 / len(pool))}
    for k in (1, 2, 3):
        v = np.zeros(len(pool))
        v[order[:k]] = new[order[:k]] if k > 1 else 1.0
        cands[f"top{k}"] = v / v.sum()
    _, scores = pool.target_diagnostics(ev_ds, ev_idx, cands, cfg.eval_batch_size, cfg.num_workers, pooled=True)
    candidate_dice = {name: scores[name] for name in cands}
    if ref_ds is not None:
        _, scores_pool = pool.target_diagnostics(ref_ds, np.arange(len(ref_ds)), cands, cfg.eval_batch_size,
                                                 cfg.num_workers, pooled=True)
        candidate_dice = {name: (1 - cfg.pool_evidence) * scores[name] + cfg.pool_evidence * scores_pool[name]
                          for name in cands}
    nsrc = {name: int((v > 1e-6).sum()) for name, v in cands.items()}
    best = max(candidate_dice.values())
    chosen_name = max((n for n in cands if candidate_dice[n] >= best - 0.005), key=lambda n: nsrc[n])
    new = cands[chosen_name]
    log("WEIGHTS", f"round {r}: labeled Dice of candidate ensembles: " + ", ".join(
        f"{n}={d:.3f}" for n, d in candidate_dice.items()) + f" -> using {chosen_name}")
    if first:
        best = int(np.argmax(dice))
        if type(pool.models[best]).__name__ == "UNet":
            d_student = student_dice(student, ev_ds, ev_idx, cfg, device)
            rec["student_dice"], rec["best_teacher_dice"] = d_student, float(dice[best])
            log("WEIGHTS", f"round {r}: labeled Dice of the student {d_student:.3f} vs best teacher "
                           f"{SHORT[pool.sources[best]]} {dice[best]:.3f}; distilling the corrected ensemble into the student")
    pool.alphas = new
    log("WEIGHTS", f"round {r}: teacher Dice on {len(idx)} labeled patches + {ev['patches'] - len(idx)} blank ({ev['fg_pixels']} fg px): "
        + ", ".join(f"{SHORT[s]}={d:.3f}" for s, d in zip(pool.sources, dice))
        + " | weights: " + ", ".join(f"{SHORT[s]}={a:.3f}" for s, a in zip(pool.sources, new)))
    rec.update({"weights": new.tolist(), "ensemble_choice": chosen_name, "candidate_dice": candidate_dice})
    return rec


def active_learning(student, train_ds, cfg, device, log, optimizer, pool, evaluate_round=None, blank=None):
    """Budget-constrained acquisition + fine-tuning loop, then consolidation; returns per-round records."""
    n_total = len(train_ds)
    budget_abs = int(cfg.budget / 100.0 * n_total)
    k = budget_abs // cfg.rounds
    if k < 1:
        raise ValueError(f"budget {cfg.budget}% of {n_total} patches gives < 1 patch per round")
    unlabeled = np.arange(n_total)
    blank_idx = None
    if blank is not None:
        blank_idx = unlabeled[blank]
        unlabeled = unlabeled[~blank]
    log("AL", f"patches={n_total} (pool {len(unlabeled)} after dropping blank) | budget={budget_abs} ({cfg.budget}%) | "
              f"rounds={cfg.rounds} | per_round={k}")
    labeled = np.zeros(0, dtype=int)
    hess = None
    records = []
    quality = {}
    consistency_set = None
    cons_scale = 1.0
    for r in range(1, cfg.rounds + 1):
        t0 = time.time()
        if len(unlabeled) == 0:
            # the pool is exhausted (budget at or above the non-blank pool): keep fine-tuning on what is labeled
            log("AL", f"round {r}/{cfg.rounds}: no unlabeled patches left ({len(labeled)} labeled); fine-tuning only")
            finetune(student, train_ds, labeled, cfg, device, log, optimizer, round_label=f"round {r}/{cfg.rounds}",
                     consistency_set=None, blank_idx=blank_idx)
            rec = {"round": r, "n_labeled": int(len(labeled)), "selection_seconds": 0.0,
                   "round_seconds": round(time.time() - t0, 1), "pool_exhausted": True}
            if evaluate_round is not None:
                rec.update(evaluate_round(student))
            records.append(rec)
            continue
        k = min(k, len(unlabeled))                 # the pool only shrinks; take what is left
        cand = unlabeled[sample_indices(len(unlabeled), cfg.score_pool_cap, cfg.seed, "cand", r)]
        h_idx = sample_indices(len(unlabeled), cfg.score_pool_cap, cfg.seed, "hess", r)
        hess = laplace_hessian(student, train_ds, unlabeled[h_idx], cfg, device)
        scores, errors, prevalence = laplace_scores(student, train_ds, cand, cfg, device, hess, with_error=(r == 1))
        if r > 1:
            # no budget on patches the student calls empty: the Laplace variance term is large on featureless tissue
            empty = prevalence < cfg.empty_prevalence
            if (~empty).sum() >= k:
                scores = np.where(empty, -np.inf, scores)
                if empty.sum():
                    log("AL", f"round {r}: {int(empty.sum())}/{len(cand)} candidates skipped as agreed-empty (predicted prevalence < {cfg.empty_prevalence})")
        if r == 1:
            quality = uncertainty_quality(scores, errors)
            if quality:
                log("AL", f"round-1 uncertainty quality on {len(cand)} patches: {quality}")
            # cold start that covers both regimes the labels must judge teachers on: half of the first batch
            # from the patches the source ensemble expects to be foreground-rich, half from foreground-poor ones
            sub = cand[sample_indices(len(cand), cfg.cold_cap, cfg.seed, "cold")]
            prev = ensemble_soft_labels(pool, train_ds, sub, cfg, device).float().mean(dim=(1, 2)).div(255.0).numpy()
            order = np.argsort(prev)
            lo, hi = sub[order[:len(sub) // 2]], sub[order[len(sub) // 2:]]
            k_hi = k - k // 2
            chosen = np.concatenate([hi[sample_indices(len(hi), k_hi, cfg.seed, "cold-hi")],
                                     lo[sample_indices(len(lo), k - k_hi, cfg.seed, "cold-lo")]])
            log("AL", f"round 1: stratified cold start: {k_hi} patches from the foreground-rich half (ensemble prevalence "
                      f">= {np.median(prev):.3f}) + {k - k_hi} from the foreground-poor half of {len(sub)} candidates")
        elif cfg.strat_bins > 0:
            chosen = cand[strat_select(scores, prevalence, k, cfg.strat_bins)]
        else:
            chosen = cand[np.argsort(-scores)[:k]]
        labeled = np.concatenate([labeled, chosen])
        unlabeled = np.setdiff1d(unlabeled, chosen, assume_unique=True)
        if consistency_set is not None:
            consistency_set.remove(chosen)
        t_sel = time.time() - t0
        student_ref = r >= 2 and bool(records) and records[-1].get("pseudo_source") == "student"
        update = update_weights(student, pool, train_ds, labeled, cfg, device, log, r, optimizer, blank_idx,
                                pool_idx=unlabeled, student_ref=student_ref)
        if update.get("first_correction") and cfg.kd_after_update > 0:
            kd_cfg = dataclasses.replace(cfg, kd_epochs=cfg.kd_after_update)
            for g in optimizer.param_groups:
                g["lr"] = cfg.kd_lr
            kd_train(student, pool, train_ds, unlabeled, kd_cfg, device, log, optimizer)
            for g in optimizer.param_groups:
                g["lr"] = cfg.ft_lr
        if cfg.consistency > 0 and (r == 1 or update.get("first_correction")):
            consistency_set = build_consistency_set(pool, train_ds, unlabeled, cfg, device, log)
        if r >= 2 and cfg.consistency > 0 and consistency_set is not None:
            # unbiased comparison on the patches just acquired (unseen by the student): who labels the pool better?
            ev_ds, ev_idx = evidence_set(train_ds, chosen, blank_idx, cfg, f"pseudo{r}")
            d_student = student_dice(student, ev_ds, ev_idx, cfg, device)
            d_ens = pool.target_diagnostics(ev_ds, ev_idx, {"ens": pool.alphas}, cfg.eval_batch_size, cfg.num_workers,
                                            pooled=True)[1]["ens"]
            update["pseudo_student_dice"], update["pseudo_ensemble_dice"] = d_student, d_ens
            prev = records[-1].get("pseudo_source") if records else None
            # the judgement set must carry real foreground (last-round leftovers can be all-empty)
            fg_ok = int(masks_fg(ev_ds, ev_idx)) >= cfg.min_evidence_fg
            if not fg_ok and prev in ("student", "ensemble"):
                use_student = prev == "student"
            elif prev == "student":
                use_student = not (d_ens > d_student + cfg.pseudo_switch_margin)      # sticky: a clear win is needed to go back
            else:
                use_student = d_student > d_ens + 0.01
            if use_student:
                consistency_set.soft = student_soft_labels(student, train_ds, consistency_set.indices, cfg, device)
                update["pseudo_source"] = "student"
            else:
                consistency_set.soft = consistency_set.ens_soft
                update["pseudo_source"] = "ensemble"
            # self-labelled consistency only as far as the student has earned it on the new labels
            g = min(1.0, max(0.0, (d_student - cfg.cons_gate_lo) / max(cfg.cons_gate_hi - cfg.cons_gate_lo, 1e-6)))
            cons_scale = g if update["pseudo_source"] == "student" else 1.0
            update["consistency_scale"] = round(cons_scale, 3)
            log("CONS", f"round {r}: on the {len(chosen)} new patches student Dice={d_student:.3f} vs ensemble {d_ens:.3f} -> pseudo-labels from the {update['pseudo_source']}"
                        + f" | consistency x{cons_scale:.2f}" + ("" if fg_ok else " | judgement set below the foreground gate: source kept"))
        finetune(student, train_ds, labeled, dataclasses.replace(cfg, consistency=cfg.consistency * cons_scale), device, log,
                 optimizer, round_label=f"round {r}/{cfg.rounds}", consistency_set=consistency_set, blank_idx=blank_idx)
        rec = {"round": r, "n_labeled": int(len(labeled)), "selection_seconds": round(t_sel, 1),
               "round_seconds": round(time.time() - t0, 1), **update}
        if evaluate_round is not None:
            rec.update(evaluate_round(student))
            log("AL", f"round {r}/{cfg.rounds} | labeled={len(labeled)} | " + " ".join(
                f"{k2}={v:.4f}" for k2, v in rec.items() if isinstance(v, float)))
        records.append(rec)
    if cfg.final_epochs > 0:
        # consolidation: all labels and pseudo-labels, cosine learning-rate decay from ft_lr to ft_lr / 10
        t0 = time.time()
        if consistency_set is not None and records[-1].get("pseudo_source") == "student":
            consistency_set.soft = student_soft_labels(student, train_ds, consistency_set.indices, cfg, device)
            log("CONS", "final consolidation: pseudo-labels refreshed from the student")
        base_lr = cfg.ft_lr
        final_cfg = dataclasses.replace(cfg, ft_epochs=1, consistency=cfg.consistency * cons_scale)
        for e in range(cfg.final_epochs):
            lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * e / max(cfg.final_epochs - 1, 1))))
            for g in optimizer.param_groups:
                g["lr"] = float(lr)
            finetune(student, train_ds, labeled, dataclasses.replace(final_cfg, seed=cfg.seed + 7919 * (e + 1)), device, log,
                     optimizer, round_label=f"final {e + 1}/{cfg.final_epochs}", consistency_set=consistency_set,
                     blank_idx=blank_idx)
        rec = {"round": cfg.rounds + 1, "n_labeled": int(len(labeled)), "selection_seconds": 0.0,
               "round_seconds": round(time.time() - t0, 1), "final_consolidation": True}
        if evaluate_round is not None:
            rec.update(evaluate_round(student))
            log("AL", f"final consolidation ({cfg.final_epochs} epochs, cosine lr) | " + " ".join(
                f"{k2}={v:.4f}" for k2, v in rec.items() if isinstance(v, float)))
        records.append(rec)
    return records, quality
