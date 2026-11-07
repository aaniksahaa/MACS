"""Optimiser and learning-rate scheduler factories for the single-source trainer."""
import torch.optim as optim

OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
SCHEDULERS = ("none", "cosine", "step", "plateau", "exponential")


def build_optimizer(name, parameters, lr, weight_decay=0.0, momentum=0.9):
    name = name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Available: {', '.join(OPTIMIZERS)}")


def build_scheduler(name, optimizer, epochs, step_size=20, gamma=0.1, patience=10, factor=0.5):
    """Schedulers are stepped once per epoch (``plateau`` on the validation loss)."""
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor)
    raise ValueError(f"Unknown scheduler '{name}'. Available: {', '.join(SCHEDULERS)}")
