#!/usr/bin/env python3
"""Train MACS on a target domain from a pool of trained source models.

    python train_macs.py --task membrane --target snemi3d --budget 10 --seed 1
    python train_macs.py --task mitochondria --target lucchi-pp --budget 5 --seeds 1,2,3
    python train_macs.py --task membrane --targets all --budgets 2,5,10 --seeds 1,2,3
    python train_macs.py --task membrane --target snemi3d --sources flywire:deeplabv3,microns_basil:resunet,whole-mouse-brain

Sources default to every other domain of the task; each source needs a checkpoint at
<models_dir>/<task>_domains/<arch>/<domain>/model_best.pth (train_source.py writes this layout).
Every run writes <output_dir>/<task>/<target>/b<budget>_s<seed>/{model.pth, result.json, rounds.csv, log.txt}
and one row of <output_dir>/results.csv; runs that already have a result.json are skipped.
Every option of macs/config.py is a command-line flag (--kd-epochs, --rounds, ...).
"""
import argparse
import dataclasses
import os

import torch

from macs import utils
from macs.config import Config
from macs.pipeline import run, run_dir


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for f in dataclasses.fields(Config):
        if f.name in ("target", "budget", "seed"):
            continue
        flag = f"--{f.name.replace('_', '-')}"
        if f.type is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=getattr(defaults, f.name))
        else:
            parser.add_argument(flag, dest=f.name, type=f.type, default=getattr(defaults, f.name))
    parser.add_argument("--target", "--targets", dest="targets", required=True,
                        help="target domain, a comma list, or 'all' (every domain of the task)")
    parser.add_argument("--budget", "--budgets", dest="budgets", default="10", help="annotation budget(s) in %%")
    parser.add_argument("--seed", "--seeds", dest="seeds", default="1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    utils.set_task(args.task, args.data_root, args.label_dir)
    fields = {f.name for f in dataclasses.fields(Config)}
    base = Config(**{k: v for k, v in vars(args).items() if k in fields})
    targets = utils.task_domains() if args.targets == "all" else [t.strip() for t in args.targets.split(",") if t.strip()]
    runs = [dataclasses.replace(base, target=utils.canonical(t), budget=float(b), seed=int(s))
            for t in targets for b in args.budgets.split(",") for s in args.seeds.split(",")]
    todo = [c for c in runs if not os.path.isfile(os.path.join(run_dir(c), "result.json"))]
    print(f"[MACS] {len(runs) - len(todo)} runs already finished, {len(todo)} to run", flush=True)
    for i, cfg in enumerate(todo, 1):
        print(f"[MACS] {i}/{len(todo)}: {cfg.target} b{cfg.budget:g} s{cfg.seed}", flush=True)
        run(cfg)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
