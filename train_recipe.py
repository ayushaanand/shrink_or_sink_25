"""
train_recipe.py
---------------
Provides the full Knowledge-Distillation training pipeline for a student model.
Imported by search.py for both proxy (20-epoch) and full (100-epoch) runs.

Key choices:
- Augmentation: RandomCrop + HorizontalFlip + ColorJitter + Normalise
- Optimizer:    AdamW (fast convergence with built-in weight decay)
- Schedule:     CosineAnnealingWarmRestarts (escapes local minima)
- Loss:         KD loss (soft targets from teacher) + hard Cross-Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import random
import numpy as np
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, ConcatDataset, Subset
import time

# ── Normalisation stats for STL-10 ────────────────────────────────────────────
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD  = [0.2603, 0.2566, 0.2713]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(96, padding=12),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders(data_root: str, teacher=None, device=None, batch_size: int = 128, top_k: int = 15000):
    """Return (train_loader, val_loader) for the standard STL-10 split."""
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # 1. Load the pristine training data setup
    train_ds = STL10(root=data_root, split="train", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = STL10(root=data_root, split="test",  download=True, transform=VAL_TRANSFORM)
    
    # 2. Extract Top-K Confident Images from the 100k Unlabeled Pool
    if teacher is not None and device is not None:
        print(f"\n[DATA] Sweeping 100k Unlabeled Dataset to find the Top {top_k:,} most mathematically confident images...")
        clean_unlab_ds = STL10(root=data_root, split="unlabeled", download=True, transform=VAL_TRANSFORM)
        eval_ld = DataLoader(clean_unlab_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
        
        teacher.eval()
        all_confs = []
        with torch.no_grad():
            for x, _ in eval_ld:
                x = x.to(device)
                probs = F.softmax(teacher(x), dim=1)
                confs, _ = torch.max(probs, dim=1)
                all_confs.append(confs.cpu())
                
        all_confs = torch.cat(all_confs)
        
        # Sort indices by highest confidence
        top_indices = torch.argsort(all_confs, descending=True)[:top_k].tolist()
        
        # Reload the unlabeled set but explicitly map it to the heavy Training Augmentations
        aug_unlab_ds = STL10(root=data_root, split="unlabeled", download=False, transform=TRAIN_TRANSFORM)
        unlab_subset = Subset(aug_unlab_ds, top_indices)
        
        combined_ds = ConcatDataset([train_ds, unlab_subset])
        print(f"[DATA] Teacher Confidence Sweep Complete!")
        print(f"[DATA] Distillation Pipeline Locked: {len(train_ds):,} Labeled + {len(unlab_subset):,} Elite Unlabeled = {len(combined_ds):,} Total Images/Epoch.\n")
    else:
        combined_ds = train_ds
        print(f"\n[DATA] Using ONLY {len(train_ds):,} Labeled images (No teacher provided).\n")
    
    # Strictly zero workers to prevent Kaggle IO/Memory Multiprocessing Deadlocks
    train_ld = DataLoader(combined_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)
    return train_ld, val_ld


# ── Knowledge Distillation Loss ────────────────────────────────────────────────

def kd_loss(student_logits, teacher_logits, hard_labels,
            temperature: float = 4.0, alpha: float = 0.7) -> torch.Tensor:
    """
    Combined Knowledge Distillation loss:
        (1 - alpha) * CrossEntropy(student, hard_labels)
      + alpha       * KL_Div(student_soft, teacher_soft)

    Args:
        temperature: Smooths the teacher's probability distribution.
                     Higher T → softer targets → more inter-class info.
        alpha:       Weight on "listening to teacher" vs ground-truth labels.
    """
    # Hard label loss
    ce = F.cross_entropy(student_logits, hard_labels, label_smoothing=0.1)

    # Soft target loss (KL divergence)
    soft_student  = F.log_softmax(student_logits  / temperature, dim=1)
    soft_teacher  = F.softmax(teacher_logits      / temperature, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    return (1.0 - alpha) * ce + alpha * kd


# ── Core train / evaluate functions ───────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, device) -> float:
    """Run one training epoch with KD. Returns mean loss."""
    student.train()
    teacher.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            t_logits = teacher(x)

        s_logits = student(x)
        
        # Inject Knowledge Distillation pseudo-labels for the Unlabeled Dataset
        hard_labels = torch.where(y == -1, t_logits.argmax(dim=1), y)
        loss = kd_loss(s_logits, t_logits, hard_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(student, loader, device) -> float:
    """Return top-1 accuracy on the given loader."""
    student.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (student(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_student(student, teacher, train_loader, val_loader,
                  epochs: int, device,
                  lr: float = 1e-3, weight_decay: float = 1e-4,
                  verbose: bool = True) -> tuple[float, list[float]]:
    """
    Full training loop for a student model under KD.

    Returns:
        final_acc: validation accuracy at the final epoch
        acc_curve: list of per-epoch val accuracies
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine Annealing with Warm Restarts:
    # T_0 = epochs // 3 cycles the LR ~3x over the full run, helping
    # the model escape local minima repeatedly.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=1
    )

    acc_curve = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss = train_one_epoch(student, teacher, train_loader, optimizer, device)
        scheduler.step()
        acc = evaluate(student, val_loader, device)
        acc_curve.append(acc)

        epoch_time = time.time() - start_time
        if verbose:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={loss:.4f}  val={acc:.4f}  ({epoch_time:.1f}s)")

    return acc_curve[-1], acc_curve
