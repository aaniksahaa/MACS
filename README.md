# MACS: Multi Domain Adaptation Enables Accurate Connectomics Segmentation

MACS segments a new electron-microscopy (EM) dataset (the *target*) by combining segmentation models trained on
several other labeled datasets (the *sources*) with a small annotation budget on the target:

1. **Source models.** One segmentation model per labeled source domain (U-Net, ResUNet or DeepLabV3; the
   architectures may differ between sources).
2. **Source weighting.** Multi-kernel maximum mean discrepancy (MK-MMD) finds the mixture of the source
   distributions closest to the target and gives one weight per source.
3. **Distillation.** A U-Net student learns from the weighted ensemble of the temperature-calibrated source
   models on unlabeled target patches.
4. **Active learning.** The annotation budget is spent over six rounds. The first batch covers patches the
   ensemble expects to be foreground-rich and foreground-poor; later batches are the patches the student is most
   uncertain about under a last-layer Laplace approximation. The labels refine the source weights and fine-tune
   the student, together with pseudo-labels on unlabeled patches; a final consolidation uses all labels.

## Installation

```bash
git clone git@github.com:aaniksahaa/MACS.git
cd MACS
conda create -n macs python=3.12 -y && conda activate macs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
pip install -r requirements.txt
```

To reproduce the exact software environment of most reported runs (Python 3.12, PyTorch 2.11, CUDA 12.8), use
`conda env create -f environment.yml` instead. Every run also records its Python, PyTorch and CUDA versions and the
code commit in `result.json`.

## Datasets

The 14 preprocessed datasets of the paper (9 membrane and 5 mitochondria domains, each split into train,
validation and test sections) are on Zenodo: <https://doi.org/10.5281/zenodo.23962155>.

```bash
bash scripts/download_datasets.sh            # -> ../datasets/labeled_datasets/{membrane_datasets,mito_datasets}
export MACS_DATASETS_ROOT=/path/to/labeled_datasets   # only if you put them elsewhere
```

Every domain folder has `{train,val,test}/raw/*.png` and the matching label folder (`membranes` or
`mitochondria`, same file names). `common/domains.py` registers the domains of both tasks:

| Task | Domains |
|---|---|
| `membrane` | `c-elegans-dauer-stage`, `dhanyasi-P14-mouse-cerebellum`, `flywire`, `katz-lab-berghia-connective`, `microns_basil`, `octopus-vulgaris-vertical-lobe-glia-deep-neuropil`, `octopus-vulgaris-vertical-lobe-sfltract`, `snemi3d`, `whole-mouse-brain` |
| `mitochondria` | `drosophila-vnc`, `fly-brain-mito-benchmark`, `kasthuri-pp`, `lucchi-pp`, `multiclass` |

`python -m common.domains --check` verifies the folders.

## 1. Train the source models

```bash
bash scripts/train_sources.sh membrane          # every membrane domain, U-Net
bash scripts/train_sources.sh mitochondria
bash scripts/train_sources.sh membrane resunet  # another architecture (resunet, deeplabv3)
```

or one model at a time:

```bash
python train_source.py --task membrane --domain snemi3d --epochs 100
python train_source.py --task mitochondria --domain lucchi-pp --epochs 100 --model deeplabv3
```

Source models are trained for 100 epochs (500 for `whole-mouse-brain`) with Adam (1e-4) and a Dice + focal + BCE
loss on 256x256 patches (stride 128), validating every 10 epochs. Checkpoints go to
`models/<task>_domains/<arch>/<domain>/`; MACS uses `model_best.pth` (best validation Dice). `python train_source.py --help`
lists every option.

## 2. Train MACS

```bash
python train_macs.py --task membrane --target snemi3d --budget 10 --seed 1
```

The sources are all other domains of the task, so this run uses the eight other membrane U-Nets and labels 10%
of the SNEMI3D training patches. Lists run one after another (finished runs are skipped):

```bash
# every target, budget and seed of the paper's main comparison
python train_macs.py --task membrane --targets all --budgets 2,5,10,20,40,60,80,100 --seeds 1,2,3
python train_macs.py --task mitochondria --targets all --budgets 2,5,10,20,40,60,80,100 --seeds 1,2,3
```

At 2% `fly-brain-mito-benchmark` has fewer labeled patches than rounds; run it with `--rounds 3`.

Choose the source pool, and mix architectures, with `--sources domain[:arch],...`:

```bash
python train_macs.py --task membrane --target dhanyasi-P14-mouse-cerebellum --budget 10 \
    --sources flywire:deeplabv3,microns_basil:resunet,snemi3d,whole-mouse-brain --run-tag mixed
```

The student is a U-Net initialised from the highest-weighted source model when that one is a U-Net
(`--init-compatible` falls back to the highest-weighted U-Net). Every hyperparameter is a flag
(`python train_macs.py --help`); the defaults in `macs/config.py` are the configuration of all reported runs.

Each run writes `outputs/<task>/<target>/b<budget>_s<seed>[_<tag>]/`:

| File | Content |
|---|---|
| `model.pth` | the trained student (U-Net state_dict) |
| `result.json` | configuration, source weights before and after refinement, per-round records, test metrics, timings |
| `rounds.csv` | labeled patches, selection time and validation Dice per round |
| `log.txt` | the full log |

and appends one line to `outputs/results.csv`. Test metrics are reported on patches (`patch_dice`, `patch_iou`,
`patch_recall`, `patch_vi`) and on whole stitched test images (`image_*`), with 8-fold test-time augmentation
(`patch_dice_notta` without it). Reusable intermediate results (features, temperatures, blank-patch masks) are
cached in `cache/macs/`.

## 3. Inference

```bash
# a split of a domain, with metrics against its labels
python predict.py --checkpoint outputs/membrane/snemi3d/b10_s1/model.pth --task membrane --domain snemi3d --out preds/snemi3d
# any folder of grayscale EM images, no labels needed
python predict.py --checkpoint outputs/membrane/snemi3d/b10_s1/model.pth --images path/to/images --out preds/mine
```

Every image gets `<name>_prob.png` (foreground probability x 255) and `<name>_mask.png` (binary, p > 0.5);
with `--domain`, `metrics.csv` and `summary.json` hold the metrics. `predict.py` also accepts a source model
checkpoint.

## Your own data and source models

Put each dataset in a folder with the layout above and pass the parent folder as `--data-root`
(`--label-dir` if the label folder is not `membranes` / `mitochondria`). Train a source model per labeled domain,
then MACS on the target:

```bash
python train_source.py --data_root my_data --task membrane --domain source_a --epochs 100
python train_source.py --data_root my_data --task membrane --domain source_b --epochs 100
python train_macs.py --data-root my_data --task membrane --target my_target --sources source_a,source_b --budget 10
```

MACS uses the target's training images without labels for the weighting and distillation, the labels of the
patches it selects within the budget, and the validation and test labels only for reporting.

## Repository structure

| Path | Content |
|---|---|
| `train_source.py` | trains one source model on one domain |
| `train_macs.py` | trains MACS on one or more targets |
| `predict.py` | segmentation (and metrics) with a trained model |
| `macs/config.py` | MACS hyperparameters (the paper's configuration) |
| `macs/pipeline.py` | one MACS run: weights, distillation, active learning, evaluation |
| `macs/weights.py` | MK-MMD mixture weights |
| `macs/teachers.py` | calibrated weighted ensemble of the source models |
| `macs/student.py` | distillation, Laplace acquisition, weight refinement, fine-tuning |
| `macs/features.py`, `macs/calibration.py`, `macs/evaluate.py`, `macs/utils.py` | features, temperature scaling, metrics, helpers |
| `archs/` | U-Net, ResUNet and DeepLabV3 |
| `common/` | dataset registry, patch dataset, losses, optimisers |
| `scripts/` | dataset download and source-model training |

## License

MIT, see [LICENSE](LICENSE).
