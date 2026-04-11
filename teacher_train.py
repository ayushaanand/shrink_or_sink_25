"""
teacher_train.py
----------------
Trains the "Ultimate Teacher" ResNet-50 on STL-10, utilizing the 100k unlabeled set.
Designed to be "unkillable" on Colab: saves comprehensive checkpoints every epoch.

Pipeline (200 Epochs):
  - Phase A (Burn-in) : Epoch 1 to 50   (Train exclusively on 5k Labeled data)
  - Phase B (Labeling): Epoch 51        (Inference on 100k Unlabeled data -> filter > 0.98 confidence)
  - Phase C (Mastery) : Epoch 51 to 200 (Train heavily on combined Augmented Labeled + High-Conf Pseudo)

Usage:
    python teacher_train.py --dataset-path ./data --out teacher_best.pth
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from PIL import Image

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="./data")
parser.add_argument("--out", type=str, default="teacher_best.pth")
parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/sos_checkpoints/teacher_latest.pth")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--burn-in", type=int, default=50)
parser.add_argument("--mastery", type=int, default=150)
parser.add_argument("--strictness", type=float, default=0.98, help="Confidence threshold for pseudo-labels")
parser.add_argument("--weights", type=str, default="", help="Path to initialized weights (for finetuning/warm restart)")
parser.add_argument("--no-download", action="store_true", help="Skip dataset download")
args = parser.parse_args()

TOTAL_EPOCHS = args.burn_in + args.mastery
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Augmentations ──────────────────────────────────────────────────────────────
# Extra heavy augmentations (AutoAugment) help prevent overfitting on noisy pseudo-labels
train_tf = transforms.Compose([
    transforms.RandomCrop(96, padding=12),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # Great for natural images like STL
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
])

# For Phase B (Inference on unlabeled data), we want pristine, unaugmented images!
clean_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
])


# ── Datasets & Loaders ─────────────────────────────────────────────────────────
train_ds = STL10(root=args.dataset_path, split="train", download=not args.no_download, transform=train_tf)
val_ds   = STL10(root=args.dataset_path, split="test",  download=not args.no_download, transform=clean_tf)
unlab_ds = STL10(root=args.dataset_path, split="unlabeled", download=not args.no_download, transform=clean_tf)

val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
unlab_ld = DataLoader(unlab_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor.tolist() # converts tensor of ints to python int list
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Helper to generate the current active training loader
def get_active_loader(epoch, pseudo_dataset=None):
    if epoch <= args.burn_in or pseudo_dataset is None:
        print(f"  [Loader] Using only {len(train_ds)} original labeled images (Burn-in).")
        return DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    else:
        combined_ds = ConcatDataset([train_ds, pseudo_dataset])
        print(f"  [Loader] Using combined {len(combined_ds)} images (Mastery Phase).")
        # Set num_workers=0 here! A huge in-memory tensor duplicated across multiple workers 
        # causes a 'copy-on-write' memory explosion and crashes Kaggle system RAM after many epochs.
        return DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)


# ── CutMix Implementation ──────────────────────────────────────────────────────
# CutMix drastically improves robustness by forcing the model to recognize object parts.
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    W, H = x.size(2), x.size(3)
    cut_rat = torch.sqrt(1. - torch.tensor(lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()

    bbx1 = torch.clamp(torch.tensor(cx - cut_w // 2), 0, W).item()
    bby1 = torch.clamp(torch.tensor(cy - cut_h // 2), 0, H).item()
    bbx2 = torch.clamp(torch.tensor(cx + cut_w // 2), 0, W).item()
    bby2 = torch.clamp(torch.tensor(cy + cut_h // 2), 0, H).item()

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[index], lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Model & State Loading ──────────────────────────────────────────────────────
# Using ResNet-50 for maximum knowledge extraction. Size doesn't matter here.
teacher = models.resnet50(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)

if args.weights and os.path.exists(args.weights):
    print(f"[WEIGHTS] Loading warm-start weights from '{args.weights}'...")
    state = torch.load(args.weights, map_location=device)
    teacher.load_state_dict(state)
    print(f"[WEIGHTS] ✅ Warm-start weights loaded successfully.")
elif args.weights:
    print(f"[WEIGHTS] ⚠️  --weights path '{args.weights}' not found. Starting from scratch.")

# ── Checkpoint State Variables ─────────────────────────────────────────────────
start_epoch = 1
best_acc = 0.0
pseudo_dataset = None
ckpt = None

# ── Load Checkpoint (The "Unkillable" Logic) ───────────────────────────────────
if os.path.exists(args.checkpoint):
    print(f"\n[RESUME] Found checkpoint at '{args.checkpoint}'. Loading...")
    print(f"[RESUME] NOTE: Checkpoint weights will OVERRIDE --weights if both are provided.")
    ckpt = torch.load(args.checkpoint, map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    print(f"[RESUME] ✅ Resumed from Epoch {ckpt['epoch']} | Best Acc so far: {best_acc:.4f}")
else:
    print(f"[RESUME] No checkpoint found at '{args.checkpoint}'. Starting fresh.")

teacher = teacher.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Training!")
    teacher = nn.DataParallel(teacher)

# ── Quick Sanity Validation ────────────────────────────────────────────────────
# Verifies loaded weights are meaningful (>80%) before wasting hours of training
if args.weights or os.path.exists(args.checkpoint):
    print("\n[SANITY] Running quick validation pass on loaded weights...")
    teacher.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            correct += (teacher(x).argmax(1) == y).sum().item()
    sanity_acc = correct / len(val_ds)
    print(f"[SANITY] Loaded weights Val Accuracy: {sanity_acc:.4f} ({sanity_acc*100:.2f}%)")
    if sanity_acc < 0.80:
        raise ValueError(f"\n🚨 ABORT: Loaded weights only achieve {sanity_acc*100:.2f}% accuracy. "
                         f"This is below the 80% safety threshold. "
                         f"The weights may be corrupted or from a poorly trained run. "
                         f"Fix --weights / --checkpoint path before wasting GPU hours!")
    print(f"[SANITY] ✅ Weights are healthy. Proceeding to training...\n")

optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if ckpt is not None:
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    
    # If we crashed during Mastery, we need to rebuild the pseudo-labels instantly
    if start_epoch > args.burn_in:
        print("  [RESUME] Restoring pseudo-labels since we are past Burn-in...")
        # (This is fast enough that we can just re-run inference instead of saving 3GB of tensors to Drive)
        pass # Rebuilt in the loop below
    print(f"  [RESUME] Resuming from Epoch {start_epoch} with Best Acc {best_acc:.4f}\n")


# ── Training Loop ──────────────────────────────────────────────────────────────
for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
    print(f"\n{'='*40}")
    print(f" Epoch {epoch:>3} / {TOTAL_EPOCHS} (Phase: {'Burn-in' if epoch <= args.burn_in else 'Mastery'})")
    
    # ── PHASE B: Generate Pseudo-Labels ──
    if epoch > args.burn_in and pseudo_dataset is None:
        print(f"\n🔍 [PHASE B] Generating strict >{args.strictness*100}% pseudo-labels on 100k unlabeled dataset...")
        teacher.eval()
        confident_indices, confident_y = [], []
        
        idx_offset = 0
        n_batches = len(unlab_ld)
        print_every = max(1, n_batches // 10)
        with torch.no_grad():
            for batch_i, (ux, _) in enumerate(unlab_ld):
                ux = ux.to(device)
                probs = F.softmax(teacher(ux), dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # Keep only predictions where probability > strictness
                mask = max_probs > args.strictness
                if mask.any():
                    batch_indices = torch.arange(idx_offset, idx_offset + len(ux))
                    confident_indices.append(batch_indices[mask.cpu()])
                    confident_y.append(preds[mask].cpu())
                
                idx_offset += len(ux)
                if (batch_i + 1) % print_every == 0 or (batch_i + 1) == n_batches:
                    pct = (batch_i + 1) / n_batches * 100
                    print(f"  [Phase B] {pct:.0f}% ({batch_i+1}/{n_batches} batches) | Confident so far: {sum(len(c) for c in confident_indices)}")
        
        if len(confident_indices) > 0:
            all_indices = torch.cat(confident_indices)
            all_y = torch.cat(confident_y)
            
            # Avoid 10GB float32 RAM explosion! Just save indices and read original uint8 data dynamically
            class DynamicLabelSubset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, indices, labels, transform):
                    self.base_data = base_dataset.data
                    self.indices = indices.tolist()
                    self.labels = labels.tolist()
                    self.transform = transform
                    
                def __len__(self):
                    return len(self.indices)
                    
                def __getitem__(self, i):
                    real_idx = self.indices[i]
                    # STL10 underlying data shape: (N, 3, 96, 96). Needs transpose for PIL
                    img = self.base_data[real_idx]
                    img = np.transpose(img, (1, 2, 0))
                    img = Image.fromarray(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, self.labels[i]

            # Apply `train_tf` so pseudo-labeled images receive heavy augmentation during mastery
            pseudo_dataset = DynamicLabelSubset(unlab_ds, all_indices, all_y, train_tf)
            print(f"✅ Successfully extracted {len(all_indices)} high-confidence images!\n")
        else:
            print("⚠ No images passed the strictness threshold. Continuing with labeled data only.\n")

    
    # Generate loader for this epoch
    active_ld = get_active_loader(epoch, pseudo_dataset)
    
    
    # ── PHASE A / C: Train ──
    teacher.train()
    running_loss = 0.0
    n_batches = len(active_ld)
    print_every = max(1, n_batches // 10)

    for batch_i, (x, y) in enumerate(active_ld):
        x, y = x.to(device), y.to(device)
        
        # Apply CutMix 50% of the time to combat pseudo-label noise
        if torch.rand(1).item() > 0.5:
            x, y_a, y_b, lam = cutmix_data(x, y)
            optimizer.zero_grad()
            logits = teacher(x)
            loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            loss = criterion(teacher(x), y)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

        if (batch_i + 1) % print_every == 0 or (batch_i + 1) == n_batches:
            pct = (batch_i + 1) / n_batches * 100
            avg = running_loss / (batch_i + 1)
            print(f"  [Train] {pct:.0f}% | batch {batch_i+1}/{n_batches} | avg_loss={avg:.4f}")

    scheduler.step()

    
    # ── Validation ──
    teacher.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            correct += (teacher(x).argmax(1) == y).sum().item()

    acc = correct / len(val_ds)
    print(f"  ↳ Loss: {running_loss/len(active_ld):.4f}  |  Val Acc: {acc:.4f} "
          f"{'⭐ NEW BEST' if acc > best_acc else ''}")

    
    # ── Robust Save ──
    save_state = teacher.module.state_dict() if isinstance(teacher, nn.DataParallel) else teacher.state_dict()
    
    if acc > best_acc:
        best_acc = acc
        # Save the final best weights specifically for the search.py
        torch.save(save_state, args.out)

    # Save the resumption checkpoint every epoch
    # We save to a temp file and rename to avoid corruption if Colab dies exactly during save
    temp_ckpt = args.checkpoint + ".tmp"
    
    # Ensure directory exists (e.g., if args.checkpoint is deeply nested)
    ckpt_dir = os.path.dirname(args.checkpoint)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state': save_state,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_acc': best_acc
    }, temp_ckpt)
    os.replace(temp_ckpt, args.checkpoint)

print(f"\n✅ All {TOTAL_EPOCHS} epochs complete. The absolute best Val Acc was {best_acc:.4f}.")
print(f"The best weights have been finalized in '{args.out}'. Let's run the search!")
