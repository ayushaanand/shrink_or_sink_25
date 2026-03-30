"""
train.py
--------
Implements the full training pipeline to faithfully reproduce the submitted model.
Leverages Knowledge Distillation from a robust ResNet-50 Teacher.
Includes RAM-caching for Kaggle T4 speedups and per-epoch accuracy reporting.

Usage:
    python train.py --dataset-path ./data --teacher-path teacher_best.pth --model-path student_final.pth
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image


from model import DynamicNet

def set_seed(seed=42):
    """Ensures deterministic, reproducible training as mandated by the rulebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """Ensures each DataLoader worker process has an independent, reproducible random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def kd_loss(logits, teacher_logits, labels, T=4.0, alpha=0.9):
    """Knowledge Distillation Loss combining hard labels and soft teacher probabilities.
    Unlabeled samples (y=-1) use the teacher's argmax as a pseudo-label for CE.
    """
    loss_kd = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1)
    ) * (T * T)
    
    # Substitute teacher pseudo-labels for unlabeled images (y == -1)
    hard_labels = torch.where(labels == -1, teacher_logits.argmax(dim=1), labels)
    loss_ce = F.cross_entropy(logits, hard_labels, label_smoothing=0.1)
    
    return (1. - alpha) * loss_ce + alpha * loss_kd

class RAMCachedSTL10(Dataset):
    """Stores all 105k uint8 images into RAM for hyperspeed distillation."""
    def __init__(self, stl10_datasets, transform=None, teacher_logits=None):
        self.transform = transform
        raw_data = np.concatenate([ds.data for ds in stl10_datasets], axis=0)
        self.data = np.transpose(raw_data, (0, 2, 3, 1))
        self.labels = np.concatenate([ds.labels for ds in stl10_datasets], axis=0)
        self.teacher_logits = teacher_logits
        
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

def get_teacher(path, device, retries=3):
    """Loads the pre-trained Ultimate Teacher (ResNet-50) robustly against network I/O failures."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n🚨 CRITICAL ERROR: The teacher weights file was not found at '{path}'. "
                                f"Please make sure you have attached the Dataset correctly in the Kaggle UI!")
        
    for attempt in range(retries):
        try:
            print(f"Loading Teacher Model from '{path}' (Attempt {attempt+1}/{retries})...")
            teacher = models.resnet50(weights=None)
            teacher.fc = nn.Linear(teacher.fc.in_features, 10)
            teacher.load_state_dict(torch.load(path, map_location=device))
            teacher = teacher.to(device)
            teacher.eval()
            print(f"✅ Teacher securely loaded from '{path}'\n")
            return teacher
        except Exception as e:
            print(f"⚠️ Warning: Failed to load teacher: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    raise RuntimeError("🚨 CRITICAL ERROR: Failed to load teacher after multiple attempts.")

def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Setting: {device}")

    # 1. Dataset Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    
    # 2. Loading Datasets
    print(f"Loading STL-10 datasets to '{args.dataset_path}'...")
    raw_train = datasets.STL10(root=args.dataset_path, split="train", download=not args.no_download, transform=test_tf)
    raw_unlab = datasets.STL10(root=args.dataset_path, split="unlabeled", download=not args.no_download, transform=test_tf)
    test_ds = datasets.STL10(root=args.dataset_path, split="test", download=not args.no_download, transform=test_tf)
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    teacher = get_teacher(args.teacher_path, device)
    
    # 3. RAM Caching & Pre-Inference (The Secret Sauce)
    print("Pre-inferencing Teacher on 105k images into RAM for hyperspeed...")
    raw_ld = DataLoader(raw_unlab, batch_size=256, shuffle=False, num_workers=2)
    teacher_logits = []
    with torch.no_grad():
        # Labeled teacher logits (precompute once)
        l_ld = DataLoader(raw_train, batch_size=256, shuffle=False, num_workers=2)
        for x, _ in l_ld: teacher_logits.append(teacher(x.to(device)).cpu())
        # Unlabeled teacher logits
        for x, _ in raw_ld: teacher_logits.append(teacher(x.to(device)).cpu())
    
    teacher_logits = torch.cat(teacher_logits, dim=0)
    
    train_cache = RAMCachedSTL10([raw_train, raw_unlab], transform=train_tf, teacher_logits=teacher_logits)
    
    g = torch.Generator()
    g.manual_seed(42)
    train_ld = DataLoader(train_cache, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    # 4. Student Setup
    print(f"Initializing Student (Widths: {args.widths}, Depths: {args.depths})...")
    student = DynamicNet(widths=args.widths, depths=args.depths).to(device)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. Resume Support
    start_epoch = 1
    if os.path.exists(args.checkpoint):
        print(f"🔄 Checkpoint found at '{args.checkpoint}'. Resuming training...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        student.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"▶️ Resuming from Epoch {start_epoch}")

    # 6. Training & Validation Loop
    print(f"Starting Training for {args.epochs} epochs with Batch Size {args.batch_size}...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        total_loss = 0.0
        
        for x, y, t_logits in train_ld:
            x, y, t_logits = x.to(device), y.to(device), t_logits.to(device)
            
            optimizer.zero_grad()
            s_logits = student(x)
            
            loss = kd_loss(s_logits, t_logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Validation Pass (Per-Epoch Accuracy)
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in test_ld:
                vx, vy = vx.to(device), vy.to(device)
                preds = student(vx).argmax(dim=1)
                correct += (preds == vy).sum().item()
                total += vy.size(0)
        
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch:03d}/{args.epochs}] | Loss: {total_loss/len(train_ld):.4f} | Val Acc: {acc:.2f}%")

        # Save Checkpoint for Auto-Resume
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint_data, args.checkpoint)

    end_time = time.time()
    print(f"\nTraining Complete in {(end_time - start_time)/60:.2f} mins!")
    print(f"Final Student Weights cast to FP16. Saving to '{args.model_path}'...")
    
    student.half()
    save_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
    
    # ── Legacy List Optimization ──
    # Saves as a [widths, depths, tensor0, ...] list to eliminate key overhead.
    # We use explicit alphabetical sorting to ensure deterministic packing/unpacking.
    keys = sorted(save_state.keys())
    weights_list = [args.widths, args.depths] + [save_state[k].cpu() for k in keys]
    torch.save(weights_list, args.model_path, _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--teacher-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="student_final.pth")
    parser.add_argument("--checkpoint", type=str, default="student_checkpoint.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--widths", type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument("--depths", type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument("--no-download", action="store_true")
    train(parser.parse_args())
