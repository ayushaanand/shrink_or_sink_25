import torch
import torch.nn as nn
from dynamic_model import DynamicNet, size_mb, param_count
from train_recipe import get_loaders, train_student
from torchvision import models
import os

import argparse

def main(args):
    print("="*65)
    print("🚀 INITIATING FINAL 25-EPOCH HACKATHON RUN")
    print("="*65)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading Teacher from '{args.teacher}'...")
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device))
    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)
    teacher = teacher.to(device).eval()

    print(f"Loading Massive 105k Dataset from '{args.data}'...")
    train_ld, val_ld = get_loaders(args.data, teacher=teacher, device=device, batch_size=128)

    cfg = [48, 96, 192, 384]
    cfg_d = [2, 3, 4, 2]
    print(f"Initializing Final Architecture: Width={cfg} Depth={cfg_d}")
    student = DynamicNet(cfg, cfg_d)
    
    mb = size_mb(student, dtype_bytes=2)
    params = param_count(student)
    print(f"Final Model Size: {mb:.4f} MB FP16 | Parameters: {params:,}")
    
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
    student = student.to(device)

    print("\nStarting 25 Epoch Distillation...")
    ckpt_name = f"final_winner_w48-96-192-384_d2-3-4-2.pth"
    acc, curve = train_student(
        student, teacher, train_ld, val_ld,
        epochs=25, device=device, lr=1e-3, verbose=True,
        ckpt_path=ckpt_name
    )
    
    print("\n" + "="*65)
    print(f"🎉 FINAL RUN COMPLETE! Test Accuracy: {acc*100:.2f}%")
    print(f"Your model is saved securely to: {ckpt_name}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/kaggle/working/data", help="Path to STL-10 dataset")
    parser.add_argument("--teacher", type=str, default="teacher_final3.pth", help="Path to Teacher .pth")
    args = parser.parse_args()
    main(args)
