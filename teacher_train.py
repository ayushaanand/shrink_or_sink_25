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
    python teacher_train.py --data ./data --out teacher_best.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./data")
parser.add_argument("--out", type=str, default="teacher_best.pth")
parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/sos_checkpoints/teacher_latest.pth")
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--burn-in", type=int, default=50)
parser.add_argument("--mastery", type=int, default=150)
parser.add_argument("--strictness", type=float, default=0.98, help="Confidence threshold for pseudo-labels")
parser.add_argument("--weights", type=str, default="", help="Path to initialized weights (for finetuning/warm restart)")
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
train_ds = STL10(root=args.data, split="train", download=True, transform=train_tf)
val_ds   = STL10(root=args.data, split="test",  download=True, transform=clean_tf)
unlab_ds = STL10(root=args.data, split="unlabeled", download=True, transform=clean_tf)

val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
unlab_ld = DataLoader(unlab_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

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
        return DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    else:
        combined_ds = ConcatDataset([train_ds, pseudo_dataset])
        print(f"  [Loader] Using combined {len(combined_ds)} images (Mastery Phase).")
        # Set num_workers=0 here! A huge in-memory tensor duplicated across multiple workers 
        # causes a 'copy-on-write' memory explosion and crashes Kaggle system RAM after many epochs.
        return DataLoader(combined_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)


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
    print(f"Loading pretrained weights from '{args.weights}' for Warm Restart...")
    teacher.load_state_dict(torch.load(args.weights, map_location=device))

# ── Checkpoint State Variables ─────────────────────────────────────────────────
start_epoch = 1
best_acc = 0.0
pseudo_dataset = None
ckpt = None

# ── Load Checkpoint (The "Unkillable" Logic) ───────────────────────────────────
if os.path.exists(args.checkpoint):
    print(f"\n[RESUME] Found checkpoint at '{args.checkpoint}'. Loading...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']

teacher = teacher.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Training!")
    teacher = nn.DataParallel(teacher)

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
        confident_x, confident_y = [], []
        
        with torch.no_grad():
            for ux, _ in tqdm(unlab_ld, desc="Inferencing"):
                ux = ux.to(device)
                probs = F.softmax(teacher(ux), dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # Keep only predictions where probability > strictness
                mask = max_probs > args.strictness
                if mask.any():
                    # Move to CPU immediately to prevent GPU memory explosion
                    confident_x.append(ux[mask].cpu())
                    confident_y.append(preds[mask].cpu())
        
        if len(confident_x) > 0:
            all_x = torch.cat(confident_x)
            all_y = torch.cat(confident_y)
            # Create a dataset but apply the training transforms so the new images get augmented during training!
            # Since all_x is already normalized by clean_tf, we need to apply spatial transforms manually,
            # or wrap it in a custom class. To save compute, we'll just train on them as-is (clean_tf).
            # The original 5k handles the augmentation.
            pseudo_dataset = PseudoDataset(all_x, all_y)
            print(f"✅ Successfully extracted {len(all_x)} high-confidence images!\n")
        else:
            print("⚠ No images passed the strictness threshold. Continuing with labeled data only.\n")

    
    # Generate loader for this epoch
    active_ld = get_active_loader(epoch, pseudo_dataset)
    
    
    # ── PHASE A / C: Train ──
    teacher.train()
    running_loss = 0.0

    for x, y in tqdm(active_ld, desc="Training", leave=False):
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
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    
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
