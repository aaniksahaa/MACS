"""MACS configuration. The defaults are the configuration of every MACS run reported in the paper."""
import dataclasses
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclasses.dataclass
class Config:
    # target, budget, sources
    task: str = "membrane"                 # membrane | mitochondria (label folder); sources default to the task's other domains
    target: str = ""
    budget: float = 10.0                   # annotation budget: % of the target's training patches that get labeled
    seed: int = 1
    sources: str = ""                      # comma list of source domains, each optionally "domain:arch" (heterogeneous pools)
    arch: str = "unet"                     # architecture of every source given without ":arch"
    run_tag: str = ""                      # optional name for this source pool; appended to the run directory
    # paths
    data_root: str = ""                    # folder with one sub-folder per domain ({train,val,test}/{raw,<label_dir>});
                                           # empty: the registry in common/domains.py (MACS_DATASETS_ROOT)
    label_dir: str = ""                    # label folder name inside every split (default: membranes / mitochondria)
    models_dir: str = os.path.join(REPO, "models")      # source models: <models_dir>/<task>_domains/<arch>/<domain>/
    teacher_ckpt: str = "model_best.pth"
    cache_dir: str = os.path.join(REPO, "cache", "macs")
    output_dir: str = os.path.join(REPO, "outputs")
    # hardware
    device: str = "cuda:0"
    num_workers: int = 8
    batch_size: int = 8
    eval_batch_size: int = 16
    amp: bool = True                       # autocast for no-grad inference only
    # MK-MMD source weighting
    feature_n: int = 512                   # patches per domain for the MMD
    feature_seed: int = 0
    source_split: str = "train"
    kernel_scales: str = "0.25,0.5,1,2,4"  # RBF bandwidths, x the median squared feature distance
    lam: float = 0.01                      # entropy coefficient of the mixture objective
    init_compatible: bool = False          # if the top-weighted teacher is not a student-shaped U-Net, initialise the
                                           # student from the highest-weighted one that is (instead of random init)
    # label-driven correction of the weights
    reliability_beta: float = 10.0         # alpha_post ~ alpha_prior * exp(beta * labeled Dice)
    update_cap: int = 512                  # labeled patches used to score the teachers
    min_evidence_fg: int = 2000            # labeled foreground pixels before labels may change the weights
    pool_evidence: float = 0.5             # blend weight of teacher agreement with the student on unlabeled patches
    pool_evidence_n: int = 2000
    kd_after_update: int = 2               # KD epochs from the corrected ensemble right after the first correction
    # distillation
    kd_epochs: int = 5
    kd_lr: float = 1e-4
    kd_cap: int = 6000                     # unlabeled patches per KD epoch
    adabn_cap: int = 512                   # unlabeled patches for the BatchNorm re-estimation after initialisation
    augment: bool = True                   # flips / 90-degree rotations during KD and fine-tuning
    # active learning
    rounds: int = 6
    cold_cap: int = 2000                   # candidates scored by the ensemble for the stratified first round
    score_pool_cap: int = 8000             # candidates scored per round (and patches for the Laplace Hessian)
    strat_bins: int = 10                   # Laplace scores are ranked inside this many predicted-foreground strata
    empty_prevalence: float = 0.005        # candidates predicted (almost) empty are skipped
    blank_std: float = 0.01                # patches with image std below this are blank tissue: free background
    # fine-tuning
    ft_epochs: int = 5
    ft_lr: float = 1e-4
    ft_cap: int = 10000                    # labeled patches per fine-tuning epoch
    ft_loss: str = "combined"              # combined (Dice + focal + BCE) | bce
    consistency: float = 0.5               # weight of the pseudo-label loss on unlabeled patches
    consistency_cap: int = 2000            # unlabeled patches with pseudo-labels
    min_consistency: int = 64              # fewer unlabeled patches left: supervised fine-tuning only
    pseudo_switch_margin: float = 0.05     # the ensemble must beat the student by this to take the pseudo-labels back
    cons_gate_lo: float = 0.5              # student pseudo-labels: consistency weight ramps from 0 at this labeled Dice ...
    cons_gate_hi: float = 0.8              # ... to full weight at this one
    final_epochs: int = 10                 # consolidation after the last round, cosine lr decay to ft_lr / 10
    # evaluation
    val_cap: int = 1024                    # validation patches for per-round monitoring
    tta: bool = True                       # 8-fold flip / rotation test-time augmentation of the final model
    image_level: bool = True               # also report metrics on stitched whole test images
