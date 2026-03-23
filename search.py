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
    python search.py --data ./data \\
                     --teacher teacher.pth \\
                     --lo  8  16  32  \\
                     --hi 64 128 256

    # 4-layer search:
    python search.py --lo  8 16  32  64 --hi 64 128 256 512
"""

import argparse
import json
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

# ── Load teacher ──────────────────────────────────────────────────────────────
teacher = models.resnet18(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher.load_state_dict(torch.load(args.teacher, map_location=device))
teacher = teacher.to(device).eval()
print(f"✅ Teacher loaded from '{args.teacher}'\n")

# ── Data loaders ──────────────────────────────────────────────────────────────
train_ld, val_ld = get_loaders(args.data, batch_size=args.batch)

# ── Interpolating Binary Search ───────────────────────────────────────────────
lo = list(args.lo)
hi = list(args.hi)

results_log = []
best_config = None
best_acc    = 0.0
best_state  = None
iteration   = 0

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
        "mb":        mb,
        "lo":        lo[:],
        "hi":        hi[:],
        "proxy_acc": proxy_acc,
        "full_acc":  None,
        "verdict":   None,
    }

    if proxy_acc < args.proxy_thresh:
        # Model can't even hit proxy threshold → too small → search higher
        print(f"  ✗ Proxy {proxy_acc:.4f} < {args.proxy_thresh} → TOO SMALL → lo = {cfg}")
        log_entry["verdict"] = "too_small_proxy"
        lo = cfg                        # raise lower bound
    else:
        # ── Phase 2: Full training ─────────────────────────────────────────
        print(f"  Phase 2: Full training ({args.full_epochs} epochs)...")
        student = DynamicNet(cfg).to(device)   # fresh weights
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
            hi = cfg                    # tighten upper bound (try smaller)
            # Track the best (smallest sufficient) model
            if best_config is None or size_mb(DynamicNet(cfg)) < size_mb(DynamicNet(best_config)):
                best_config = cfg
                best_acc    = full_acc
                best_state  = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        else:
            print(f"  ✗ {full_acc:.4f} < {args.target_acc} — INSUFFICIENT → lo = {cfg}")
            log_entry["verdict"] = "insufficient"
            lo = cfg                    # raise lower bound (need bigger)

    results_log.append(log_entry)

# ── Report ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Search Converged")
print(f"  Final bounds:  lo={lo}  hi={hi}")
print("=" * 65)

if best_config:
    mb = size_mb(DynamicNet(best_config))
    print(f"\n  🏆 Winner config : {best_config}")
    print(f"     Val accuracy  : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"     Est. size     : {mb:.4f} MB")

    student = DynamicNet(best_config)
    student.load_state_dict(best_state)
    torch.save(student.state_dict(), args.out)
    print(f"     Saved to      : {args.out}")
else:
    print("\n  ⚠ No config met the target accuracy within these bounds.")
    print("    Try widening --hi or adjusting --target-acc.")

# Save results log
log_path = "search_results.json"
with open(log_path, "w") as f:
    json.dump(results_log, f, indent=2)
print(f"\n📋 Full results log → '{log_path}'")
