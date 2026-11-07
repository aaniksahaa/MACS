"""Temperature scaling of teachers and calibration metrics on pixel probabilities."""
import numpy as np
import torch
import torch.nn.functional as F

from .utils import make_loader


@torch.no_grad()
def collect_pixels(model, dataset, indices, device, pixels_per_patch=2048, batch_size=16,
                   num_workers=8, seed=0, temperature=1.0):
    """Random pixel sample of (logit, label) over the given patches; returns two 1-D tensors on CPU."""
    gen = torch.Generator().manual_seed(seed)
    logits, labels = [], []
    model.eval()
    for imgs, masks in make_loader(dataset, indices, batch_size, num_workers):
        out = model(imgs.to(device, non_blocking=True)).float().cpu() / temperature
        b, _, h, w = out.shape
        flat_out = out.reshape(b, -1)
        flat_mask = masks.reshape(b, -1)
        pick = torch.randint(0, h * w, (b, min(pixels_per_patch, h * w)), generator=gen)
        logits.append(torch.gather(flat_out, 1, pick).reshape(-1))
        labels.append(torch.gather(flat_mask, 1, pick).reshape(-1))
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels, max_iter=200):
    """Minimise NLL over T > 0 (parameterised as log T), LBFGS on the pixel sample."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    # Bound T: a two-image validation split can push the fit to extremes; outside [0.25, 8] the
    # teacher is reported as miscalibrated and kept at the bound.
    return float(min(max(log_t.exp().item(), 0.25), 8.0))


def calibration_metrics(probs, labels, bins=15):
    """ECE (confidence of the predicted class), NLL, Brier, mean confidence, prevalences."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=np.float64)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5) == (y >= 0.5)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return {"ece": float(ece), "nll": nll, "brier": float(((p - y) ** 2).mean()),
            "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()),
            "predicted_prevalence": float((p >= 0.5).mean()), "true_prevalence": float((y >= 0.5).mean()),
            "mean_probability": float(p.mean())}
