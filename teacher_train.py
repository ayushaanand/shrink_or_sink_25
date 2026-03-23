"""
teacher_train.py
----------------
Trains a large, unconstrained ResNet-18 teacher model on STL-10.
The teacher's size doesn't matter — only the tiny student is submitted.
The saved teacher weights are used by search.py for Knowledge Distillation.

Usage:
    python teacher_train.py --data ./data --epochs 100 --out teacher.pth
"""
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",   type=str, default="./data")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--batch",  type=int, default=128)
parser.add_argument("--out",    type=str, default="teacher.pth")
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Augmentations ──────────────────────────────────────────────────────────────
# Strong augmentations help the teacher generalise on STL-10's small labeled set.
train_tf = transforms.Compose([
    transforms.RandomCrop(96, padding=12),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                         std=[0.2603, 0.2566, 0.2713]),
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                         std=[0.2603, 0.2566, 0.2713]),
])

train_ds = STL10(root=args.data, split="train", download=True, transform=train_tf)
val_ds   = STL10(root=args.data, split="test",  download=True, transform=val_tf)

train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                      num_workers=2, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                      num_workers=2, pin_memory=True)

# ── Teacher Model (ResNet-18, trained from scratch on STL-10) ──────────────────
# We are NOT loading ImageNet weights — that is against the rules.
teacher = models.resnet18(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher = teacher.to(device)

# ── Optimiser + Scheduler ──────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Training Loop ──────────────────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(1, args.epochs + 1):
    teacher.train()
    running_loss = 0.0

    for x, y in tqdm(train_ld, desc=f"[Teacher] Epoch {epoch}/{args.epochs}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(teacher(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # ── Validation ──────────────────────────────────────────────────────────
    teacher.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            correct += (teacher(x).argmax(1) == y).sum().item()

    acc = correct / len(val_ds)
    print(f"Epoch {epoch:>3}  loss={running_loss/len(train_ld):.4f}  val_acc={acc:.4f}"
          f"{'  ← best' if acc > best_acc else ''}")

    if acc > best_acc:
        best_acc = acc
        torch.save(teacher.state_dict(), args.out)

print(f"\n✅ Teacher training done. Best val acc: {best_acc:.4f}. Saved to '{args.out}'.")
