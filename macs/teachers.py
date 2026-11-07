"""The pool of source models: loading, calibrated ensemble predictions, target diagnostics."""
import numpy as np
import torch

from .utils import load_teacher, make_loader
from .evaluate import patch_dice


class TeacherPool:
    def __init__(self, models_dir, sources, arch, device, which="model_best.pth"):
        """sources: domain names; arch: one architecture for all, or a list with one architecture per source
        (heterogeneous pools; the same domain may appear with several architectures)."""
        self.sources = list(sources)
        self.archs = list(arch) if isinstance(arch, (list, tuple)) else [arch] * len(self.sources)
        self.arch = self.archs[0] if len(set(self.archs)) == 1 else "mixed"
        self.device = device
        self.models, self.configs, self.paths = [], [], []
        for domain, a in zip(self.sources, self.archs):
            model, config, path = load_teacher(models_dir, domain, a, device, which)
            self.models.append(model)
            self.configs.append(config)
            self.paths.append(path)
        self.temperatures = np.ones(len(self.sources))
        self.alphas = np.full(len(self.sources), 1.0 / len(self.sources))

    def __len__(self):
        return len(self.models)

    @torch.no_grad()
    def logits(self, i, x):
        return self.models[i](x).float() / float(self.temperatures[i])

    @torch.no_grad()
    def ensemble(self, x, alphas=None):
        """Weighted average of the calibrated teacher probabilities for a batch on the device."""
        alphas = self.alphas if alphas is None else alphas
        acc = None
        for i, a in enumerate(alphas):
            if a <= 0:
                continue
            term = a * torch.sigmoid(self.logits(i, x))
            acc = term if acc is None else acc + term
        return acc

    @torch.no_grad()
    def teacher_dice(self, dataset, indices, batch_size=16, num_workers=8, pooled=False):
        """Dice of every calibrated teacher on the given labeled patches (patch mean, or pooled pixels)."""
        return self.target_diagnostics(dataset, indices, {}, batch_size, num_workers, pooled)[0]

    @torch.no_grad()
    def target_diagnostics(self, dataset, indices, weight_sets, batch_size=16, num_workers=8, pooled=False):
        """Dice of every teacher (calibrated) and of every ensemble in weight_sets (calibrated probability average,
        what KD distils). pooled=True pools the pixels of all patches (2 TP / (|pred| + |mask|)) instead of averaging per-patch
        Dice, which a handful of labeled patches can estimate and which empty patches cannot dominate.
        self.evidence records the labeled foreground pixels and non-empty patches of the last call.
        Returns (per_teacher, {"<set>": dice})."""
        per_teacher = [[] for _ in self.models]
        per_set = {}
        acc = {}                                     # pooled: key -> [intersection, |pred|, |mask|]
        fg, nonempty = 0, 0
        temps = torch.as_tensor(self.temperatures, dtype=torch.float32, device=self.device).view(-1, 1, 1, 1, 1)

        def take(key, prob):
            if pooled:
                pred = (prob > 0.5).float()
                a = acc.setdefault(key, [0.0, 0.0, 0.0])
                a[0] += float((pred * masks).sum()); a[1] += float(pred.sum()); a[2] += float(masks.sum())
            elif isinstance(key, int):
                per_teacher[key].extend(patch_dice(prob, masks).tolist())
            else:
                per_set.setdefault(key, []).extend(patch_dice(prob, masks).tolist())

        for imgs, masks in make_loader(dataset, indices, batch_size, num_workers):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            fg += int(masks.sum().item())
            nonempty += int((masks.flatten(1).sum(1) > 0).sum().item())
            raw = torch.stack([self.models[i](imgs).float() for i in range(len(self.models))])   # [m, B, 1, H, W]
            cal = raw / temps
            for i in range(len(self.models)):
                take(i, torch.sigmoid(cal[i]))
            for name, alphas in weight_sets.items():
                a = torch.as_tensor(np.asarray(alphas), dtype=raw.dtype, device=raw.device).view(-1, 1, 1, 1, 1)
                take(name, (torch.sigmoid(cal) * a).sum(0))
        self.evidence = {"fg_pixels": fg, "nonempty_patches": nonempty,
                         "patches": len(indices) if indices is not None else len(dataset)}
        if pooled:
            dice = {k: (2 * v[0] + 1.0) / (v[1] + v[2] + 1.0) for k, v in acc.items()}
            return ([dice[i] for i in range(len(self.models))], {k: v for k, v in dice.items() if not isinstance(k, int)})
        return ([float(np.mean(v)) for v in per_teacher], {k: float(np.mean(v)) for k, v in per_set.items()})
