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
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
import time
import tqdm
import torchvision.datasets.utils as tv_utils
from PIL import Image

# ── 🚨 Hotfix: Suppress Rapid TQDM Output for Kaggle Disconnects ──────────────
class MinimalTqdm:
    def __init__(self, *args, **kwargs):
        self.iterable = args[0] if args else kwargs.get('iterable')
        self.total = kwargs.get('total')
        self.n = 0
        self.last_pct = -1
    def __iter__(self):
        if self.iterable is not None:
            for obj in self.iterable:
                yield obj
                self.update(1)
    def update(self, n=1):
        self.n += n
        if self.total:
            pct = int((self.n / self.total) * 10) * 10
            if pct > self.last_pct:
                print(f"  [STL-10 Download] {pct}% Complete...")
                self.last_pct = pct
    def set_postfix(self, **kwargs): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Physically override PyTorch's native tqdm fetcher globally
tv_utils.tqdm = MinimalTqdm
tqdm.tqdm = MinimalTqdm
# ──────────────────────────────────────────────────────────────────────────────

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

class RAMCachedSTL10(Dataset):
    """Stores 105k uint8 STL-10 JPEGs entirely in RAM (2.9 GB), bypassing physical hard drives."""
    def __init__(self, stl10_datasets, transform=None):
        self.transform = transform
        # Pre-transpose to (N, H, W, C) so fromarray() is faster in the workers
        raw_data = np.concatenate([ds.data for ds in stl10_datasets], axis=0) 
        self.data = np.transpose(raw_data, (0, 2, 3, 1)) 
        self.labels = np.concatenate([ds.labels for ds in stl10_datasets], axis=0)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img = self.data[index]
        target = int(self.labels[index])
        
        # Directly convert pre-transposed uint8 to PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_loaders(data_root: str, teacher=None, device=None, batch_size: int = 256):
    """Return (train_loader, val_loader) using the massive 105k Distillation pool with safe Kaggle workers."""
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # ── Safe Check for Kaggle Read-Only Cloud Filesystem ───────────────────────
    download_flag = True
    if "kaggle/input" in data_root.replace("\\", "/").lower():
        download_flag = False

    train_ds = STL10(root=data_root, split="train", download=download_flag)
    unlab_ds = STL10(root=data_root, split="unlabeled", download=download_flag)
    val_ds   = STL10(root=data_root, split="test",  download=download_flag, transform=VAL_TRANSFORM)
    
    print(f"\n[RAM CACHE] Loading {len(train_ds)+len(unlab_ds):,} JPEGs natively into physical UInt8 RAM (2.9 GB)...")
    combined_ds = RAMCachedSTL10([train_ds, unlab_ds], transform=TRAIN_TRANSFORM)
    print(f"[DATA] Restored Massive Distillation Pipeline: Total {len(combined_ds):,} Images/Epoch.\n")
    
    # Unleashed Multiprocessing: 4 Workers + RAM Cache + No Pinning Deadlocks
    train_ld = DataLoader(combined_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=False, worker_init_fn=seed_worker, generator=g)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=False)
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
                  verbose: bool = True,
                  proxy_epochs: int = None, proxy_thresh: float = None,
                  ckpt_path: str = None,
                  resume_state: dict = None, active_ckpt_path: str = None,
                  cfg: list = None, cfg_d: list = None, search_state: dict = None) -> tuple[float, list[float]]:
    """
    Full integrated training loop for a student model under KD.
    Supports in-line proxy checking and per-epoch check-pointing to survive Kaggle crashes.
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine Annealing with Warm Restarts:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=1
    )

    acc_curve = []
    start_epoch = 1

    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        acc_curve = resume_state['acc_curve']
        start_epoch = resume_state['epoch'] + 1
        if verbose:
            print(f"  [RESUME] Flawlessly reviving Momentum Buffers & LR from Epoch {start_epoch}...")

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss = train_one_epoch(student, teacher, train_loader, optimizer, device)
        scheduler.step()
        acc = evaluate(student, val_loader, device)
        acc_curve.append(acc)

        epoch_time = time.time() - start_time
        if verbose:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={loss:.4f}  val={acc:.4f}  ({epoch_time:.1f}s)")
            
        if active_ckpt_path and cfg is not None and cfg_d is not None:
            heavy_state = {
                'epoch': epoch,
                'model': student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc_curve': acc_curve,
                'cfg': cfg,
                'cfg_d': cfg_d,
                'search_state': search_state
            }
            torch.save(heavy_state, active_ckpt_path)

        if ckpt_path:
            raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            fp16_state = {k: v.cpu().clone().half() for k, v in raw_state.items()}
            torch.save(fp16_state, ckpt_path)

        if proxy_epochs is not None and proxy_thresh is not None:
            if epoch == proxy_epochs:
                if acc < proxy_thresh:
                    if verbose:
                        print(f"  ✗ Proxy threshold failed: {acc:.4f} < {proxy_thresh:.4f}. Ejecting early!")
                    return acc, acc_curve

    return acc_curve[-1], acc_curve
