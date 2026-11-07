"""Soft registry of segmentation tasks and their domains.

Nothing is copied into the repository: a domain is a *name* that points at a
dataset directory inside the shared ``labeled_datasets`` tree, plus the name of
the label folder for the task. Every dataset directory has the layout

    <dataset>/{train,val,test}/raw/<file>.png
    <dataset>/{train,val,test}/<label_dir>/<file>.png     (same file names)

Two tasks are defined: ``membrane`` (label folder ``membranes``) and
``mitochondria`` (label folder ``mitochondria``). A task only ever reads the
label folder it is registered for.

Checkpoints follow one layout, built by :func:`model_dir`::

    <models_dir>/<task>_domains/<model>/<domain>/model_best.pth

The dataset tree defaults to ``../datasets/labeled_datasets`` next to this
repository; set ``MACS_DATASETS_ROOT`` to point elsewhere. A flat
``<data_root>/<domain>`` layout is also accepted through an explicit ``data_root``.

    python -m common.domains                       # every task and domain with its path
    python -m common.domains --task mitochondria   # one task
    python -m common.domains --check               # verify directories, counts, label values
    python -m common.domains --model-dirs --models unet,resunet --models-dir models
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASETS_ROOT = os.path.normpath(os.path.join(REPO_DIR, "..", "datasets", "labeled_datasets"))
DATASETS_ROOT = os.environ.get("MACS_DATASETS_ROOT", DEFAULT_DATASETS_ROOT)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Domain:
    name: str
    path: str                       # relative to DATASETS_ROOT
    ignore_label: int = None        # mask value marking "no tissue"; see common/data.py
    notes: str = ""


@dataclass(frozen=True)
class Task:
    name: str
    label_dir: str                  # label folder inside every split
    domains: dict = field(default_factory=dict)


def _task(name, label_dir, domains):
    return Task(name, label_dir, {d.name: d for d in domains})


TASKS = {
    "membrane": _task("membrane", "membranes", [
        Domain("c-elegans-dauer-stage", "membrane_datasets/c-elegans-dauer-stage"),
        Domain("dhanyasi-P14-mouse-cerebellum", "membrane_datasets/dhanyasi-P14-mouse-cerebellum"),
        Domain("flywire", "membrane_datasets/flywire"),
        Domain("katz-lab-berghia-connective", "membrane_datasets/katz-lab-berghia-connective"),
        Domain("microns_basil", "membrane_datasets/microns_basil"),
        Domain("octopus-vulgaris-vertical-lobe-glia-deep-neuropil",
               "membrane_datasets/octopus-vulgaris-vertical-lobe-glia-deep-neuropil"),
        Domain("octopus-vulgaris-vertical-lobe-sfltract",
               "membrane_datasets/octopus-vulgaris-vertical-lobe-sfltract"),
        Domain("snemi3d", "membrane_datasets/snemi3d"),
        Domain("whole-mouse-brain", "membrane_datasets/whole-mouse-brain"),
    ]),
    "mitochondria": _task("mitochondria", "mitochondria", [
        Domain("fly-brain-mito-benchmark", "mito_datasets/fly-brain-mito-benchmark",
               notes="255x255 slices: one pixel of reflect padding gives one 256 patch per slice"),
        Domain("kasthuri-pp", "mito_datasets/kasthuri-pp", ignore_label=2,
               notes="mask value 2 (raw 0) is padding outside the rotated tissue; "
                     "patches that are padding only are dropped, the rest counts as background"),
        Domain("lucchi-pp", "mito_datasets/lucchi-pp"),
        Domain("drosophila-vnc", "mito_datasets/drosophila-vnc",
               notes="anti-aliased 8-bit mitochondria masks (thresholded at 0.5)"),
        Domain("multiclass", "mito_datasets/multiclass",
               notes="EPFL hippocampus volume; anti-aliased 8-bit mitochondria masks (thresholded at 0.5)"),
    ]),
}

# Short names accepted for three membrane domains.
ALIASES = {
    "fly": "flywire",
    "micron": "microns_basil",
    "snemi": "snemi3d",
}


# ------------------------------------------------------------------ lookups
def list_tasks():
    return list(TASKS)


def get_task(task):
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Available: {', '.join(TASKS)}")
    return TASKS[task]


def list_domains(task):
    return list(get_task(task).domains)


def canonical_domain(name):
    return ALIASES.get(name, name)


def infer_task(domain):
    """The task a domain belongs to; an error if the name is unknown or ambiguous."""
    name = canonical_domain(domain)
    owners = [task for task, spec in TASKS.items() if name in spec.domains]
    if not owners:
        known = ", ".join(f"{t}: {', '.join(s.domains)}" for t, s in TASKS.items())
        raise KeyError(f"Unknown domain '{domain}'. Known domains -- {known}")
    if len(owners) > 1:
        raise KeyError(f"Domain '{domain}' belongs to several tasks ({', '.join(owners)}); pass --task")
    return owners[0]


def resolve(domain, task=None, allow_unknown=False):
    """``(task, Task, Domain)`` for a domain name, accepting aliases and inferring the task.

    With ``allow_unknown`` (``data_root`` layouts) a name outside the registry is
    returned as a bare ``Domain`` of ``task`` (default ``membrane``) instead of an error.
    """
    name = canonical_domain(domain)
    if task is None:
        try:
            task = infer_task(name)
        except KeyError:
            if not allow_unknown:
                raise
            task = "membrane"
    spec = get_task(task)
    if name not in spec.domains:
        if not allow_unknown:
            raise KeyError(f"Domain '{domain}' is not a {task} domain. Available: {', '.join(spec.domains)}")
        return task, spec, Domain(name, path=None, notes="not in common/domains.py; data_root layout")
    return task, spec, spec.domains[name]


def label_dir(task):
    return get_task(task).label_dir


def domain_path(domain, task=None, data_root=None, datasets_root=None):
    """Directory holding ``{train,val,test}`` for a domain.

    ``data_root`` selects the flat layout ``<data_root>/<domain>`` (the name is
    used as given, so ``data/fly`` keeps working); otherwise the registry path under
    ``datasets_root`` (default ``DATASETS_ROOT``) is returned.
    """
    if data_root:
        return os.path.join(data_root, domain)
    _, _, spec = resolve(domain, task)
    return os.path.join(datasets_root or DATASETS_ROOT, spec.path)


def model_dir(models_dir, task, domain, model, allow_unknown=False):
    """``<models_dir>/<task>_domains/<model>/<domain>``."""
    task, spec, dom = resolve(domain, task, allow_unknown)
    return os.path.join(models_dir, f"{task}_domains", str(model).lower(), dom.name)


# ------------------------------------------------------------------ checks
def check_domain(domain, task=None, data_root=None, scan_labels=False):
    """Verify a domain directory: splits, raw/label pairing, optionally label values.

    Returns a dict with ``ok``, ``problems`` and per-split counts; with ``scan_labels``
    every mask is read and its distinct values and image sizes are collected.
    """
    task, spec, dom = resolve(domain, task)
    root = domain_path(domain, task, data_root)
    result = {"task": task, "domain": dom.name, "path": root, "label_dir": spec.label_dir,
              "splits": {}, "problems": []}
    if not os.path.isdir(root):
        result["problems"].append(f"missing directory: {root}")
    for split in SPLITS:
        raw_dir = os.path.join(root, split, "raw")
        mask_dir = os.path.join(root, split, spec.label_dir)
        info = {"raw": None, "labels": None}
        if not os.path.isdir(raw_dir):
            result["problems"].append(f"missing {raw_dir}")
        if not os.path.isdir(mask_dir):
            result["problems"].append(f"missing {mask_dir}")
        if os.path.isdir(raw_dir) and os.path.isdir(mask_dir):
            raw = sorted(os.listdir(raw_dir))
            masks = sorted(os.listdir(mask_dir))
            info.update(raw=len(raw), labels=len(masks))
            if not raw:
                result["problems"].append(f"empty {raw_dir}")
            if raw != masks:
                missing = sorted(set(raw) ^ set(masks))[:3]
                result["problems"].append(
                    f"{split}: raw and {spec.label_dir} file names differ (e.g. {missing})")
            if scan_labels:
                import numpy as np
                from PIL import Image
                values, sizes = set(), set()
                for fname in masks:
                    with Image.open(os.path.join(mask_dir, fname)) as mask:
                        values.update(np.unique(np.asarray(mask.convert("L"))).tolist())
                        sizes.add(mask.size)
                info.update(label_values=sorted(values), sizes=sorted(sizes))
        result["splits"][split] = info
    result["ok"] = not result["problems"]
    return result


# ------------------------------------------------------------------ CLI
def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, choices=list_tasks(), help="restrict to one task")
    parser.add_argument("--domains", default=None, help="comma-separated domain names (default: all)")
    parser.add_argument("--data-root", default=None, help="flat layout <data_root>/<domain>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--names", action="store_true", help="print 'task<TAB>domain' only")
    mode.add_argument("--check", action="store_true", help="verify directories and file pairing")
    mode.add_argument("--scan", action="store_true", help="--check plus every mask's values and sizes")
    mode.add_argument("--model-dirs", action="store_true",
                      help="print 'task<TAB>domain<TAB>model<TAB>dir' for --models under --models-dir")
    parser.add_argument("--models", default="unet", help="comma-separated architectures for --model-dirs")
    parser.add_argument("--models-dir", default="models", help="checkpoint root for --model-dirs")
    args = parser.parse_args(argv)

    tasks = [args.task] if args.task else list_tasks()
    wanted = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
    pairs = []
    for task in tasks:
        for name in list_domains(task):
            if wanted is None or name in wanted or any(canonical_domain(w) == name for w in wanted):
                pairs.append((task, name))
    if wanted:
        # A name from another task is simply not selected here;
        # a name no task knows is an error.
        all_names = {name for spec in TASKS.values() for name in spec.domains}
        unknown = [w for w in wanted if canonical_domain(w) not in all_names]
        if unknown:
            print(f"Unknown domain(s): {', '.join(unknown)} (python -m common.domains --names)", file=sys.stderr)
            return 2

    failed = False
    for task, name in pairs:
        if args.names:
            print(f"{task}\t{name}")
        elif args.model_dirs:
            for model in [m.strip() for m in args.models.split(",") if m.strip()]:
                print(f"{task}\t{name}\t{model}\t{model_dir(args.models_dir, task, name, model)}")
        elif args.check or args.scan:
            report = check_domain(name, task, args.data_root, scan_labels=args.scan)
            status = "ok" if report["ok"] else "PROBLEM"
            counts = " ".join(f"{s}={i['raw']}" for s, i in report["splits"].items())
            print(f"[{status}] {task:12s} {name:50s} {counts}  {report['path']}")
            for problem in report["problems"]:
                print(f"         - {problem}")
            if args.scan:
                for split, info in report["splits"].items():
                    if "label_values" in info:
                        vals = info["label_values"]
                        shown = vals if len(vals) <= 6 else [vals[0], vals[1], "...", vals[-2], vals[-1]]
                        print(f"         {split}: label_values={shown} sizes={info['sizes'][:4]}")
            failed |= not report["ok"]
        else:
            spec = TASKS[task]
            print(f"{task:12s} {name:50s} {spec.label_dir:12s} {domain_path(name, task, args.data_root)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
