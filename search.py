"""
search.py
---------
Interpolating Binary Search over DynamicNet configurations.

Instead of scanning a fixed list, we maintain a (lo_config, hi_config) pair
and binary-search by probing their element-wise midpoint each iteration.

Example progression (3-layer search):
    lo=[8,16,32]  hi=[64,128,256]  → probe [36,72,144]
    → if ≥85%: hi=[36,72,144]   → probe [22,44,88]
    → if <85%:  lo=[22,44,88]   → probe [29,58,116]
    ... until hi - lo ≤ tol in every dimension.

Usage:
    python search.py --data ./data \
                     --teacher teacher.pth \
                     --lo  8  16  32  \
                     --hi 64 128 256

    # 4-layer search:
    python search.py --lo  8 16  32  64 --hi 64 128 256 256
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
from torchvision import models

from dynamic_model import DynamicNet, size_mb, param_count, midpoint, configs_converged
from train_recipe import get_loaders, train_student

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",          type=str,   default="./data")
parser.add_argument("--teacher",       type=str,   default="teacher.pth",
                    help="Path to pre-trained teacher .pth")
parser.add_argument("--lo",            type=int,   nargs="+", default=[8, 16, 32],
                    help="Lower-bound channel config (must be same length as --hi)")
parser.add_argument("--hi",            type=int,   nargs="+", default=[64, 128, 256],
                    help="Upper-bound channel config")
parser.add_argument("--tol",           type=int,   default=2,
                    help="Stop when every channel dimension differs by ≤ tol")
parser.add_argument("--step",          type=int,   default=2,
                    help="Round midpoint channels to nearest multiple of this value")
parser.add_argument("--proxy-epochs",  type=int,   default=20)
parser.add_argument("--full-epochs",   type=int,   default=100)
parser.add_argument("--proxy-thresh",  type=float, default=0.65,
                    help="Min val acc at proxy-epochs to proceed to full training")
parser.add_argument("--target-acc",    type=float, default=0.85)
parser.add_argument("--batch",         type=int,   default=128)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--teacher-min-acc", type=float, default=0.82,
                    help="Minimum teacher val acc required to proceed")
parser.add_argument("--proxy-ratio",   type=float, default=0.85,
                    help="Proxy threshold = proxy_ratio * hi's accuracy at proxy_epochs. "
                         "Smaller values give small models more benefit of the doubt.")
parser.add_argument("--out",           type=str,   default="best_student.pth")
args = parser.parse_args()

assert len(args.lo) == len(args.hi), \
    f"--lo and --hi must have the same number of elements. Got {args.lo} vs {args.hi}"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── Validate that hi config itself fits under 5 MB ────────────────────────────
hi_model = DynamicNet(args.hi)
assert size_mb(hi_model) < 5.0, (
    f"--hi config {args.hi} is already {size_mb(hi_model):.3f} MB — exceeds 5 MB limit!"
)
print(f"Search bounds:")
print(f"  lo = {args.lo}  ({size_mb(DynamicNet(args.lo)):.4f} MB)")
print(f"  hi = {args.hi}  ({size_mb(DynamicNet(args.hi)):.4f} MB)\n")

# ── Load teacher securely ─────────────────────────────────────────────────────
def load_teacher_safe(path, device, retries=3):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n🚨 CRITICAL ERROR: The teacher weights file was not found at '{path}'. "
                                f"Please make sure you have attached the Dataset correctly in the Kaggle UI!")
        
    for attempt in range(retries):
        try:
            print(f"Loading Teacher Model from '{path}' (Attempt {attempt+1}/{retries})...")
            # Fixed critical bug: Previously ResNet-18, changed to correct ResNet-50
            t_model = models.resnet50(weights=None)
            t_model.fc = nn.Linear(t_model.fc.in_features, 10)
            t_model.load_state_dict(torch.load(path, map_location=device))
            t_model = t_model.to(device).eval()
            print(f"✅ Teacher securely loaded from '{path}'\n")
            return t_model
        except Exception as e:
            print(f"⚠️ Warning: Failed to load teacher: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    raise RuntimeError("🚨 CRITICAL ERROR: Failed to load teacher after multiple attempts. The file might be corrupted.")

teacher = load_teacher_safe(args.teacher, device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Teacher inference!")
    teacher = nn.DataParallel(teacher)

# ── Data loaders ──────────────────────────────────────────────────────────────
train_ld, val_ld = get_loaders(args.data, batch_size=args.batch)

# ── Verifying Teacher Quality ─────────────────────────────────────────────────
print("Evaluating loaded teacher model on validation set to ensure quality...")
correct = 0
total = 0
with torch.no_grad():
    for x, y in val_ld:
        x, y = x.to(device), y.to(device)
        out = teacher(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        
teacher_acc = correct / total
print(f"Teacher Validation Accuracy: {teacher_acc:.4f}  ({teacher_acc*100:.2f}%)")

if teacher_acc < args.teacher_min_acc:
    raise ValueError(f"\n🚨 CRITICAL ERROR: The loaded Teacher only reached {teacher_acc*100:.2f}% accuracy. "
                     f"This is strictly below your required {args.teacher_min_acc*100:.2f}% limit. "
                     f"Distilling from a poorly trained teacher will cripple your student model permanently. Aborting search.")
print("✅ Teacher quality verified. Proceeding to student search...\n")

# ── Pre-validate hi: calibrate proxy threshold + confirm hi reaches target ─────────
print("=" * 65)
print(f"Pre-validating upper bound hi={args.hi}...")
print(f"  Step 1: {args.proxy_epochs} proxy epochs to calibrate dynamic threshold")
print(f"  Step 2: remaining {args.full_epochs - args.proxy_epochs} epochs to confirm ≥{args.target_acc*100:.0f}%")
print("=" * 65)

_hi_student = DynamicNet(args.hi).to(device)
if torch.cuda.device_count() > 1:
    _hi_student = nn.DataParallel(_hi_student)

# Step 1: proxy run — get hi's epoch-N accuracy as calibration baseline
_, _hi_proxy_curve = train_student(
    _hi_student, teacher, train_ld, val_ld,
    epochs=args.proxy_epochs, device=device, lr=args.lr, verbose=False,
)
hi_proxy_acc = max(_hi_proxy_curve)
dynamic_proxy_thresh = round(hi_proxy_acc * args.proxy_ratio, 4)
print(f"  hi proxy acc  @ epoch {args.proxy_epochs}: {hi_proxy_acc:.4f}")
print(f"  proxy_ratio   : {args.proxy_ratio}")
print(f"  → dynamic_proxy_thresh = {dynamic_proxy_thresh:.4f}  "
      f"({dynamic_proxy_thresh*100:.2f}% required at epoch {args.proxy_epochs} to proceed)")

# Step 2: continue hi training to full_epochs to confirm it can hit target
_hi_acc, _ = train_student(
    _hi_student, teacher, train_ld, val_ld,
    epochs=args.full_epochs - args.proxy_epochs, device=device, lr=args.lr, verbose=False,
)
print(f"  hi full acc   @ epoch {args.full_epochs}: {_hi_acc:.4f} ({_hi_acc*100:.2f}%)")
if _hi_acc < args.target_acc:
    raise ValueError(
        f"\n🚨 ABORT: Upper bound hi={args.hi} only reached {_hi_acc*100:.2f}% — "
        f"below your {args.target_acc*100:.0f}% target even after full training. "
        f"Widen --hi or lower --target-acc!"
    )
print(f"\u2705 hi confirmed sufficient ({_hi_acc:.4f} ≥ {args.target_acc}). Starting search!\n")
del _hi_student  # free VRAM before search begins

# ── Interpolating Binary Search ───────────────────────────────────────────────
lo = list(args.lo)
hi = list(args.hi)

results_log    = []
# Best-by-SIZE: smallest model that still hits target_acc
best_config    = None
best_acc       = 0.0
best_state     = None
# Best-by-ACC: highest accuracy model seen regardless of size
best_acc_config = None
best_acc_val    = 0.0
best_acc_state  = None
iteration      = 0

print("=" * 65)
print("Starting Interpolating Binary Search")
print("=" * 65)


while not configs_converged(lo, hi, tol=args.tol):
    iteration += 1
    cfg = midpoint(lo, hi, step=args.step)

    mb = size_mb(DynamicNet(cfg))
    params = param_count(DynamicNet(cfg))
    print(f"\n[Iter {iteration}]  Probing {cfg}  "
          f"({params:,} params  {mb:.4f} MB)")
    print(f"  lo={lo}  hi={hi}")
    print("─" * 65)

    student = DynamicNet(cfg).to(device)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)

    # ── Phase 1: Proxy training ────────────────────────────────────────────
    print(f"  Phase 1: Proxy ({args.proxy_epochs} epochs)...")
    _, acc_curve = train_student(
        student, teacher, train_ld, val_ld,
        epochs=args.proxy_epochs,
        device=device,
        lr=args.lr,
        verbose=True,
    )
    proxy_acc = max(acc_curve)
    print(f"  Proxy best acc: {proxy_acc:.4f}")

    log_entry = {
        "iteration": iteration,
        "config":    cfg,
        "params":    params,
        "mb":        round(mb, 4),
        "lo":        lo[:],
        "hi":        hi[:],
        "proxy_acc": proxy_acc,
        "full_acc":  None,
        "verdict":   None,
    }

    if proxy_acc < dynamic_proxy_thresh:
        # Model can't even hit proxy threshold → too small → search higher
        print(f"  ✗ Proxy {proxy_acc:.4f} < {dynamic_proxy_thresh:.4f} (={args.proxy_ratio}×hi) → TOO SMALL → lo = {cfg}")
        log_entry["verdict"] = "too_small_proxy"
        if cfg == lo:
            lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
        else:
            lo = cfg
    else:
        # ── Phase 2: Full training ─────────────────────────────────────────
        print(f"  Phase 2: Full training ({args.full_epochs} epochs)...")
        student = DynamicNet(cfg).to(device)   # fresh weights
        if torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        full_acc, _ = train_student(
            student, teacher, train_ld, val_ld,
            epochs=args.full_epochs,
            device=device,
            lr=args.lr,
            verbose=True,
        )
        log_entry["full_acc"] = full_acc
        print(f"  Full best acc:  {full_acc:.4f}")

        if full_acc >= args.target_acc:
            print(f"  ✓ {full_acc:.4f} ≥ {args.target_acc} — SUFFICIENT → hi = {cfg}")
            log_entry["verdict"] = "sufficient"
            if cfg == hi:
                hi = [h - args.step if h > l else h for l, h in zip(lo, hi)]
            else:
                hi = cfg
            # Track the best-by-SIZE (smallest sufficient model)
            if best_config is None or size_mb(DynamicNet(cfg)) < size_mb(DynamicNet(best_config)):
                best_config = cfg
                best_acc    = full_acc
                raw_state   = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
                best_state  = {k: v.cpu().clone() for k, v in raw_state.items()}
        else:
            print(f"  ✗ {full_acc:.4f} < {args.target_acc} — INSUFFICIENT → lo = {cfg}")
            log_entry["verdict"] = "insufficient"
            if cfg == lo:
                lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
            else:
                lo = cfg

        # Track best-by-ACCURACY (highest accuracy seen, regardless of size)
        if full_acc > best_acc_val:
            best_acc_val    = full_acc
            best_acc_config = cfg
            raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            best_acc_state = {k: v.cpu().clone() for k, v in raw_state.items()}
            # Save immediately so a crash doesn't lose this
            _tmp = DynamicNet(cfg)
            _tmp.load_state_dict(best_acc_state)
            torch.save(_tmp.state_dict(), args.out + ".best_acc.pth")
            print(f"  🥇 New accuracy record! {full_acc:.4f} — saved to '{args.out}.best_acc.pth'")

        # Print live leaderboard
        print(f"\n  ── Live Leaderboard ──")
        if best_config:
            print(f"     Best-by-SIZE : {best_config}  |  {size_mb(DynamicNet(best_config)):.4f} MB  |  acc={best_acc:.4f}")
        print(f"     Best-by-ACC  : {best_acc_config}  |  {size_mb(DynamicNet(best_acc_config)) if best_acc_config else 0:.4f} MB  |  acc={best_acc_val:.4f}")

    results_log.append(log_entry)

# ── Final Report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Search Converged")
print(f"  Final bounds:  lo={lo}  hi={hi}")
print("=" * 65)

if best_config:
    mb = size_mb(DynamicNet(best_config))
    print(f"\n  🏆 Winner (Smallest passing model):")
    print(f"     Config       : {best_config}")
    print(f"     Params       : {param_count(DynamicNet(best_config)):,}")
    print(f"     Size         : {mb:.4f} MB")
    print(f"     Val Accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    student = DynamicNet(best_config)
    student.load_state_dict(best_state)
    torch.save(student.state_dict(), args.out)
    print(f"     Saved to     : '{args.out}'")
else:
    print("\n  ⚠ No config met the target accuracy within these bounds.")
    print("    Try widening --hi or adjusting --target-acc.")

if best_acc_config:
    mb2 = size_mb(DynamicNet(best_acc_config))
    print(f"\n  🥇 Best Accuracy Model (may be larger):")
    print(f"     Config       : {best_acc_config}")
    print(f"     Params       : {param_count(DynamicNet(best_acc_config)):,}")
    print(f"     Size         : {mb2:.4f} MB")
    print(f"     Val Accuracy : {best_acc_val:.4f}  ({best_acc_val*100:.2f}%)")
    print(f"     Saved to     : '{args.out}.best_acc.pth'  (already on disk)")

# Save full results log
log_path = "search_results.json"
with open(log_path, "w") as f:
    json.dump(results_log, f, indent=2)
print(f"\n📋 Full results log → '{log_path}'")
