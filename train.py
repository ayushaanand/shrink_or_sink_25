"""
train.py
--------
Implements the full training pipeline to faithfully reproduce the submitted model.
Leverages Knowledge Distillation from a robust ResNet-50 Teacher.

Usage:
    python train.py --dataset-path ./data --teacher-path teacher_best.pth --out final_submission.pth
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DynamicNet

def set_seed(seed=42):
    """Ensures deterministic, reproducible training as mandated by the rulebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def kd_loss(logits, teacher_logits, labels, T=4.0, alpha=0.9):
    """Knowledge Distillation Loss combining hard labels and soft teacher probabilities."""
    loss_ce = F.cross_entropy(logits, labels)
    loss_kd = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1)
    ) * (T * T)
    return (1. - alpha) * loss_ce + alpha * loss_kd

def get_teacher(path, device):
    """Loads the pre-trained Ultimate Teacher (ResNet-50) for distillation."""
    print(f"Loading Teacher Model from '{path}'...")
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    teacher.load_state_dict(torch.load(path, map_location=device))
    teacher = teacher.to(device)
    teacher.eval()
    return teacher

def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # STL-10 augmentations (ColorJitter + RandCrop/Flip)
    train_tf = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])
    
    print(f"Loading STL-10 training set to '{args.dataset_path}'...")
    train_ds = datasets.STL10(root=args.dataset_path, split="train", download=True, transform=train_tf)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    teacher = get_teacher(args.teacher_path, device)
    
    print("Initializing tiny Student Model from model.py...")
    student = DynamicNet().to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Starting Knowledge Distillation for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        
        for x, y in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                t_logits = teacher(x)
                
            optimizer.zero_grad()
            s_logits = student(x)
            
            loss = kd_loss(s_logits, t_logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_ld):.4f}")

    print(f"\nTraining Complete! Saving final weights to '{args.out}'...")
    torch.save(student.state_dict(), args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path for STL-10 dataset")
    parser.add_argument("--teacher-path", type=str, required=True, help="Path to teacher weights")
    parser.add_argument("--out", type=str, default="final_submission.pth", help="Output filename")
    parser.add_argument("--epochs", type=int, default=100, help="Training length")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    train(parser.parse_args())
