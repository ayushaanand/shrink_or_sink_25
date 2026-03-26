"""
search.py
---------
Interpolating Binary Search over DynamicNet configurations, now supporting 2D search (Width & Depth).

Usage:
    python search.py --data ./data \
                     --teacher teacher.pth \
                     --lo  8  16  32  --lo-depth 1 1 1 \
                     --hi 64 128 256  --hi-depth 3 3 3
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
from torchvision import models

from dynamic_model import DynamicNet, size_mb, param_count, midpoint, midpoint_depth, configs_converged
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
parser.add_argument("--lo-depth",      type=int,   nargs="+", default=[1, 1, 1],
                    help="Lower-bound depth config")
parser.add_argument("--hi-depth",      type=int,   nargs="+", default=[3, 3, 3],
                    help="Upper-bound depth config")
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
                    help="Proxy threshold = proxy_ratio * hi's accuracy at proxy_epochs. ")
parser.add_argument("--direct-thresh", type=float, default=None,
                    help="Skip Phase 1 'hi' pre-validation and force this dynamic threshold.")
parser.add_argument("--out",           type=str,   default="best_student.pth")
args = parser.parse_args()

assert len(args.lo) == len(args.hi) == len(args.lo_depth) == len(args.hi_depth), \
    f"All bound configs (--lo, --hi, --lo-depth, --hi-depth) must have the exact same number of elements."

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── Validate that hi config itself fits under 5 MB ────────────────────────────
# NOTE: size_mb uses FP16 internally if dtype_bytes=2!
hi_model = DynamicNet(args.hi, args.hi_depth)
assert size_mb(hi_model, dtype_bytes=2) < 5.0, (
    f"--hi config is already {size_mb(hi_model, 2):.3f} MB (FP16) — exceeds 5 MB limit!"
)
print(f"Search bounds:")
print(f"  lo = w:{args.lo} d:{args.lo_depth}  ({size_mb(DynamicNet(args.lo, args.lo_depth), 2):.4f} MB FP16)")
print(f"  hi = w:{args.hi} d:{args.hi_depth}  ({size_mb(DynamicNet(args.hi, args.hi_depth), 2):.4f} MB FP16)\n")

# ── Load teacher securely ─────────────────────────────────────────────────────
def load_teacher_safe(path, device, retries=3):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n🚨 CRITICAL ERROR: The teacher weights file was not found at '{path}'. ")
        
    for attempt in range(retries):
        try:
            print(f"Loading Teacher Model from '{path}' (Attempt {attempt+1}/{retries})...")
            t_model = models.resnet50(weights=None)
            t_model.fc = nn.Linear(t_model.fc.in_features, 10)
            t_model.load_state_dict(torch.load(path, map_location=device))
            t_model = t_model.to(device).eval()
            print(f"✅ Teacher securely loaded from '{path}'\n")
            return t_model
        except Exception as e:
            print(f"⚠️ Warning: Failed to load teacher: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    raise RuntimeError("🚨 CRITICAL ERROR: Failed to load teacher after multiple attempts.")

teacher = load_teacher_safe(args.teacher, device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Teacher inference!")
    teacher = nn.DataParallel(teacher)

# ── Data loaders ──────────────────────────────────────────────────────────────
train_ld, val_ld = get_loaders(args.data, teacher=teacher, device=device, batch_size=args.batch)

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
                     f"This is strictly below your required {args.teacher_min_acc*100:.2f}%. Aborting.")
print("✅ Teacher quality verified. Proceeding to student search...\n")

# ── Pre-validate hi: calibrate proxy threshold + confirm hi reaches target ─────────
if args.direct_thresh is not None:
    dynamic_proxy_thresh = args.direct_thresh
    print("=" * 65)
    print(f"[STRATEGY] Skipping 'hi' pre-validation!")
    print(f"Hardcoding dynamic_proxy_thresh = {dynamic_proxy_thresh:.4f}")
    print("=" * 65)
else:
    print("=" * 65)
    print(f"Pre-validating upper bound hi_w={args.hi} hi_d={args.hi_depth}...")
    print(f"  Step 1: {args.proxy_epochs} proxy epochs to calibrate dynamic threshold")
    print(f"  Step 2: remaining {args.full_epochs - args.proxy_epochs} epochs to confirm ≥{args.target_acc*100:.0f}% target")
    print("=" * 65)

    _hi_student = DynamicNet(args.hi, args.hi_depth).to(device)
    if torch.cuda.device_count() > 1:
        _hi_student = nn.DataParallel(_hi_student)

    # Step 1: proxy run — VERBOSE enabled as requested by user!
    _, _hi_proxy_curve = train_student(
        _hi_student, teacher, train_ld, val_ld,
        epochs=args.proxy_epochs, device=device, lr=args.lr, verbose=True,
    )
    hi_proxy_acc = _hi_proxy_curve[-1]
    dynamic_proxy_thresh = round(hi_proxy_acc * args.proxy_ratio, 4)
    print(f"  hi proxy acc  @ epoch {args.proxy_epochs}: {hi_proxy_acc:.4f}")
    print(f"  → dynamic_proxy_thresh = {dynamic_proxy_thresh:.4f}  "
          f"({dynamic_proxy_thresh*100:.2f}% required at epoch {args.proxy_epochs} to proceed)\n")

    # Step 2: continue hi training to full_epochs to confirm hitting target
    _hi_acc, _ = train_student(
        _hi_student, teacher, train_ld, val_ld,
        epochs=args.full_epochs - args.proxy_epochs, device=device, lr=args.lr, verbose=True,
    )
    print(f"  hi full acc   @ epoch {args.full_epochs}: {_hi_acc:.4f} ({_hi_acc*100:.2f}%)")
    if _hi_acc < args.target_acc:
        raise ValueError(
            f"\n🚨 ABORT: Upper bound hi_w={args.hi} hi_d={args.hi_depth} only reached {_hi_acc*100:.2f}% "
            f"— below your {args.target_acc*100:.0f}% target. Widen --hi/--hi-depth or lower --target-acc!"
        )
    print(f"\u2705 hi confirmed sufficient ({_hi_acc:.4f} ≥ {args.target_acc}). Starting search!\n")
    del _hi_student

# ── Interpolating Binary Search ───────────────────────────────────────────────
lo = list(args.lo)
hi = list(args.hi)
lo_d = list(args.lo_depth)
hi_d = list(args.hi_depth)

results_log    = []
best_config    = None
best_d_config  = None
best_acc       = 0.0
best_state     = None
best_acc_config = None
best_acc_d_config = None
best_acc_val    = 0.0
best_acc_state  = None
iteration      = 0

print("=" * 65)
print("Starting 2D Interpolating Binary Search")
print("=" * 65)

# Converged when both width and depth converge within their tolerances.
# Width uses user tol (e.g., 2 channels), Depth uses strict 1 layer tol.
def check_convergence(l_w, h_w, l_d, h_d):
    w_conv = configs_converged(l_w, h_w, tol=args.tol)
    d_conv = configs_converged(l_d, h_d, tol=1)
    return w_conv and d_conv

evaluated_lo = False

while not check_convergence(lo, hi, lo_d, hi_d):
    iteration += 1
    
    if not evaluated_lo:
        cfg, cfg_d = list(lo), list(lo_d)
        evaluated_lo = True
        print("\n  [STRATEGY] Evaluating pure 'lo' bound first to check instant victory!")
    else:
        cfg = midpoint(lo, hi, step=args.step)
        cfg_d = midpoint_depth(lo_d, hi_d)

    mb = size_mb(DynamicNet(cfg, cfg_d), dtype_bytes=2) # FP16 sizing
    params = param_count(DynamicNet(cfg, cfg_d))
    
    print(f"\n[Iter {iteration}]  Probing w_cfg={cfg} d_cfg={cfg_d}  "
          f"({params:,} params  {mb:.4f} MB FP16)")
    print(f"  lo_w={lo}  hi_w={hi}    lo_d={lo_d}  hi_d={hi_d}")
    print("─" * 65)

    student = DynamicNet(cfg, cfg_d).to(device)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)

    # ── Phase 1: Proxy training ────────────────────────────────────────────
    print(f"  Phase 1: Proxy ({args.proxy_epochs} epochs)...")
    _, acc_curve = train_student(
        student, teacher, train_ld, val_ld,
        epochs=args.proxy_epochs, device=device, lr=args.lr, verbose=True,
    )
    proxy_acc = acc_curve[-1]
    print(f"  Proxy best acc: {proxy_acc:.4f}")

    log_entry = {
        "iteration": iteration,
        "config_w":  cfg,
        "config_d":  cfg_d,
        "params":    params,
        "mb_fp16":   round(mb, 4),
        "proxy_acc": proxy_acc,
        "full_acc":  None,
        "verdict":   None,
    }

    if proxy_acc < dynamic_proxy_thresh:
        print(f"  ✗ Proxy {proxy_acc:.4f} < {dynamic_proxy_thresh:.4f} → TOO SMALL → lo = current")
        log_entry["verdict"] = "too_small_proxy"
        if cfg == lo and cfg_d == lo_d:
            # force bump if completely stuck
            lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
            lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
        else:
            lo, lo_d = cfg, cfg_d
    else:
        # ── Phase 2: Full training ─────────────────────────────────────────
        print(f"  Phase 2: Full training ({args.full_epochs} epochs)...")
        student = DynamicNet(cfg, cfg_d).to(device)   # fresh weights
        if torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
            
        full_acc, _ = train_student(
            student, teacher, train_ld, val_ld,
            epochs=args.full_epochs, device=device, lr=args.lr, verbose=True,
        )
        log_entry["full_acc"] = full_acc
        print(f"  Full best acc:  {full_acc:.4f}")
        
        # Save this exact configuration regardless of whether it wins or loses
        raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
        w_str = "-".join(map(str, cfg))
        d_str = "-".join(map(str, cfg_d))
        ckpt_name = f"student_w{w_str}_d{d_str}_acc{full_acc:.4f}.pth"
        _tmp = DynamicNet(cfg, cfg_d)
        _tmp.load_state_dict({k: v.cpu().clone() for k, v in raw_state.items()})
        torch.save(_tmp.half().state_dict(), ckpt_name)
        print(f"  [SAVE] Intermediate sweep checkpoint secured: '{ckpt_name}'")

        if full_acc >= args.target_acc:
            print(f"  ✓ {full_acc:.4f} ≥ {args.target_acc} — SUFFICIENT → hi = current")
            log_entry["verdict"] = "sufficient"
            if cfg == hi and cfg_d == hi_d:
                hi = [h - args.step if h > l else h for l, h in zip(lo, hi)]
                hi_d = [hd - 1 if hd > ld else hd for ld, hd in zip(lo_d, hi_d)]
            else:
                hi, hi_d = cfg, cfg_d
                
            # Track best-by-SIZE
            if best_config is None or size_mb(DynamicNet(cfg, cfg_d)) < size_mb(DynamicNet(best_config, best_d_config)):
                best_config, best_d_config = cfg, cfg_d
                best_acc = full_acc
                raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
                best_state = {k: v.cpu().clone() for k, v in raw_state.items()}
        else:
            print(f"  ✗ {full_acc:.4f} < {args.target_acc} — INSUFFICIENT → lo = current")
            log_entry["verdict"] = "insufficient"
            if cfg == lo and cfg_d == lo_d:
                lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
                lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
            else:
                lo, lo_d = cfg, cfg_d

        # Track best-by-ACCURACY
        if full_acc > best_acc_val:
            best_acc_val = full_acc
            best_acc_config, best_acc_d_config = cfg, cfg_d
            raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            best_acc_state = {k: v.cpu().clone() for k, v in raw_state.items()}
            
            _tmp = DynamicNet(cfg, cfg_d)
            _tmp.load_state_dict(best_acc_state)
            # CAUTION: Saving as FP16
            torch.save(_tmp.half().state_dict(), args.out + ".best_acc.pth")
            print(f"  🥇 New accuracy record! {full_acc:.4f} — saved FP16 to '{args.out}.best_acc.pth'")

        print(f"\n  ── Live Leaderboard ──")
        if best_config:
            print(f"     Best-by-SIZE : w={best_config} d={best_d_config} | acc={best_acc:.4f}")
        print(f"     Best-by-ACC  : w={best_acc_config} d={best_acc_d_config} | acc={best_acc_val:.4f}")

    results_log.append(log_entry)

# ── Final Report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Search Converged")
print(f"  Final bounds: w: lo={lo} hi={hi} | d: lo={lo_d} hi={hi_d}")
print("=" * 65)

if best_config:
    mb = size_mb(DynamicNet(best_config, best_d_config), dtype_bytes=2)
    print(f"\n  🏆 Winner (Smallest passing model):")
    print(f"     Config Width : {best_config}")
    print(f"     Config Depth : {best_d_config}")
    print(f"     Params       : {param_count(DynamicNet(best_config, best_d_config)):,}")
    print(f"     Size (FP16)  : {mb:.4f} MB")
    print(f"     Val Accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    
    student = DynamicNet(best_config, best_d_config)
    student.load_state_dict(best_state)
    # CAUTION: Saving as FP16
    torch.save(student.half().state_dict(), args.out)
    print(f"     Saved to     : '{args.out}'")
else:
    print("\n  ⚠ No config met the target accuracy within these bounds.")

if best_acc_config:
    mb2 = size_mb(DynamicNet(best_acc_config, best_acc_d_config), dtype_bytes=2)
    print(f"\n  🥇 Best Accuracy Model (may be larger):")
    print(f"     Config Width : {best_acc_config}")
    print(f"     Config Depth : {best_acc_d_config}")
    print(f"     Size (FP16)  : {mb2:.4f} MB")
    print(f"     Val Accuracy : {best_acc_val:.4f}  ({best_acc_val*100:.2f}%)")
    print(f"     Saved to     : '{args.out}.best_acc.pth'  (already on disk)")

with open("search_results.json", "w") as f:
    json.dump(results_log, f, indent=2)
