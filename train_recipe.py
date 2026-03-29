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

# CPU workers: crop + flip + color jitter (no hue, too slow) + normalize
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(96, padding=12),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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
    """Stores 105k uint8 STL-10 images in RAM + optional pre-computed teacher logits."""
    def __init__(self, stl10_datasets, transform=None, teacher_logits=None):
        self.transform = transform
        raw_data = np.concatenate([ds.data for ds in stl10_datasets], axis=0)
        self.data = np.transpose(raw_data, (0, 2, 3, 1))  # (N, H, W, C) pre-transposed
        self.labels = np.concatenate([ds.labels for ds in stl10_datasets], axis=0)
        self.teacher_logits = teacher_logits  # (N, 10) FP16 tensor, or None
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.labels[index])
        if self.teacher_logits is not None:
            return img, target, self.teacher_logits[index]
        return img, target

def get_loaders(data_root: str, teacher=None, device=None, batch_size: int = 256):
    """Return (train_loader, val_loader). Teacher logits are pre-computed once and cached in RAM."""
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    # ── Safe Check for Kaggle Read-Only Cloud Filesystem ───────────────────────
    download_flag = True
    if "kaggle/input" in data_root.replace("\\", "/").lower():
        download_flag = False

    train_ds = STL10(root=data_root, split="train",     download=download_flag)
    unlab_ds = STL10(root=data_root, split="unlabeled", download=download_flag)
    val_ds   = STL10(root=data_root, split="test",      download=download_flag, transform=VAL_TRANSFORM)

    # ── Pre-compute teacher logits once for ALL 105k training images ────────────
    teacher_logits = None
    if teacher is not None and device is not None:
        print(f"[LOGIT CACHE] Pre-computing teacher logits for {len(train_ds)+len(unlab_ds):,} images (runs once)...")

        # Minimal no-augmentation dataset purely for inference
        raw_chw = np.concatenate([train_ds.data, unlab_ds.data], axis=0)  # (N, 3, 96, 96) uint8
        _norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(STL10_MEAN, STL10_STD)])

        class _InferDs(Dataset):
            def __init__(self, chw): self.d = np.transpose(chw, (0, 2, 3, 1))
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return _norm(Image.fromarray(self.d[i]))

        infer_ld = DataLoader(_InferDs(raw_chw), batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=False)
        teacher.eval()
        all_logits = []
        with torch.no_grad():
            for x in infer_ld:
                all_logits.append(teacher(x.to(device)).cpu().half())
        teacher_logits = torch.cat(all_logits, dim=0)  # (N, 10) FP16
        print(f"[LOGIT CACHE] Done! {len(teacher_logits):,} logits cached ({teacher_logits.nbytes / 1e6:.1f} MB). Teacher will NOT run during training.\n")

    print(f"[RAM CACHE] Loading {len(train_ds)+len(unlab_ds):,} images into RAM...")
    combined_ds = RAMCachedSTL10([train_ds, unlab_ds], transform=TRAIN_TRANSFORM, teacher_logits=teacher_logits)
    print(f"[DATA] {len(combined_ds):,} images/epoch ready.\n")

    train_ld = DataLoader(combined_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=False, worker_init_fn=seed_worker, generator=g)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=False)
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
    """Run one training epoch with KD. Uses pre-cached teacher logits if available."""
    student.train()
    total_loss = 0.0

    for batch in loader:
        if len(batch) == 3:
            x, y, t_logits = batch
            x, y = x.to(device), y.to(device)
            t_logits = t_logits.to(device).float()
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            teacher.eval()
            with torch.no_grad():
                t_logits = teacher(x)

        optimizer.zero_grad()
        s_logits = student(x)
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
                  total_epochs: int = None, target_acc: float = None,
                  ckpt_path: str = None,
                  resume_state: dict = None, active_ckpt_path: str = None,
                  cfg: list = None, cfg_d: list = None, search_state: dict = None) -> tuple[float, list[float]]:
    """
    Full integrated training loop for a student model under KD.
    Supports in-line proxy checking and per-epoch check-pointing to survive Kaggle crashes.
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    # Single cosine decay over the full training window — no restarts that could
    # spike the LR on the final epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    acc_curve = []
    start_epoch = 1

    if resume_state is not None and 'optimizer' in resume_state:
        if isinstance(student, nn.DataParallel):
            student.module.load_state_dict(resume_state['model'])
        else:
            student.load_state_dict(resume_state['model'])
            
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        acc_curve = resume_state.get('acc_curve', [])
        start_epoch = resume_state.get('epoch', 0) + 1
        if verbose:
            print(f"  [RESUME] Flawlessly reviving Model Weights, Buffers & LR from Epoch {start_epoch}...")

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

        if proxy_epochs is not None and epoch == proxy_epochs:
            do_cut = False
            if target_acc is not None and total_epochs is not None:
                # Trajectory projection: linear extrapolation from slope at proxy
                window = max(3, proxy_epochs // 5)
                if len(acc_curve) > window:
                    slope = (acc_curve[-1] - acc_curve[-window - 1]) / window
                else:
                    slope = 0.0
                projected = acc_curve[-1] + slope * (total_epochs - proxy_epochs)
                if projected < target_acc * 0.92:
                    if verbose:
                        print(f"  ✗ Trajectory cut: acc={acc_curve[-1]:.4f} "
                              f"slope={slope:+.5f}/ep "
                              f"projected@{total_epochs}={projected:.4f} "
                              f"< {target_acc * 0.92:.4f}. Ejecting.")
                    do_cut = True
            elif proxy_thresh is not None and acc < proxy_thresh:
                # Fallback flat threshold (backward compat)
                if verbose:
                    print(f"  ✗ Proxy threshold: {acc:.4f} < {proxy_thresh:.4f}. Ejecting.")
                do_cut = True
            if do_cut:
                return acc, acc_curve

    return acc_curve[-1], acc_curve
