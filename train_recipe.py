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
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def get_loaders(data_root: str, batch_size: int = 128):
    """Return (train_loader, val_loader) for the standard STL-10 split."""
    train_ds = STL10(root=data_root, split="train", download=True, transform=TRAIN_TRANSFORM)
    val_ds   = STL10(root=data_root, split="test",  download=True, transform=VAL_TRANSFORM)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
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
        loss = kd_loss(s_logits, t_logits, y)
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
        best_acc: highest val accuracy achieved
        acc_curve: list of per-epoch val accuracies (for proxy checks)
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine Annealing with Warm Restarts:
    # T_0 = epochs // 3 cycles the LR ~3x over the full run, helping
    # the model escape local minima repeatedly.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=1
    )

    best_acc = 0.0
    acc_curve = []
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(student, teacher, train_loader, optimizer, device)
        scheduler.step()
        acc = evaluate(student, val_loader, device)
        acc_curve.append(acc)

        marker = ""
        if acc > best_acc:
            best_acc = acc
            raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            best_state = {k: v.cpu().clone() for k, v in raw_state.items()}
            marker = "  ← best"

        if verbose:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={loss:.4f}  val={acc:.4f}{marker}")

    # Restore best weights into the student
    # If wrapped in DataParallel, load into the inner .module (no `module.` prefix in best_state)
    if best_state is not None:
        target = student.module if isinstance(student, nn.DataParallel) else student
        target.load_state_dict(best_state)

    return best_acc, acc_curve
