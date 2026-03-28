"""
search.py
---------
2D Binary Search over DynamicNet (Width × Depth) with:
  - Variable epoch budget: smaller models get more epochs (fairer comparison)
  - Trajectory-projection proxy: cut hopeless models at the midpoint epoch
  - MB-gap convergence: stop when hi_mb - lo_mb < --size-tol-mb
  - Immortal per-epoch checkpointing for Kaggle timeout survival
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

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",            type=str,   default="./data")
parser.add_argument("--teacher",         type=str,   default="teacher.pth")
parser.add_argument("--lo",              type=int,   nargs="+", default=[8, 16, 32])
parser.add_argument("--hi",              type=int,   nargs="+", default=[64, 128, 256])
parser.add_argument("--lo-depth",        type=int,   nargs="+", default=[1, 1, 1])
parser.add_argument("--hi-depth",        type=int,   nargs="+", default=[3, 3, 3])
parser.add_argument("--tol",             type=int,   default=2,
                    help="Channel-level fallback convergence tolerance")
parser.add_argument("--step",            type=int,   default=2,
                    help="Round midpoint channels to nearest multiple of this value")
parser.add_argument("--full-epochs",     type=int,   default=40,
                    help="Base epochs for the hi model. Smaller models get more via --epoch-scale")
parser.add_argument("--max-epochs",      type=int,   default=90,
                    help="Hard cap on epochs granted to any single node")
parser.add_argument("--epoch-scale",     type=float, default=0.45,
                    help="Exponent: node_epochs = full_epochs * (hi_mb/cfg_mb)^epoch_scale")
parser.add_argument("--size-tol-mb",     type=float, default=0.04,
                    help="Stop search when hi_mb - lo_mb falls below this (MB, FP16)")
parser.add_argument("--target-acc",      type=float, default=0.85)
parser.add_argument("--batch",           type=int,   default=512)
parser.add_argument("--lr",              type=float, default=1e-3)
parser.add_argument("--teacher-min-acc", type=float, default=0.82)
parser.add_argument("--skip-hi-val",     action="store_true",
                    help="Skip hi model validation (use when hi is already known to be valid)")
parser.add_argument("--out",             type=str,   default="best_student.pth")
args = parser.parse_args()


def main():
    assert len(args.lo) == len(args.hi) == len(args.lo_depth) == len(args.hi_depth), \
        "All bound configs must have the same number of stages."

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Validate hi config stays under 5 MB ───────────────────────────────────
    HI_MB = size_mb(DynamicNet(args.hi, args.hi_depth), dtype_bytes=2)
    assert HI_MB < 5.0, f"--hi config is {HI_MB:.3f} MB FP16 — exceeds 5 MB limit!"

    print("Search bounds:")
    print(f"  lo = w:{args.lo} d:{args.lo_depth}  ({size_mb(DynamicNet(args.lo, args.lo_depth), 2):.4f} MB FP16)")
    print(f"  hi = w:{args.hi} d:{args.hi_depth}  ({HI_MB:.4f} MB FP16)")
    print(f"  Convergence: stop when hi_mb - lo_mb < {args.size_tol_mb} MB\n")

    # ── Resume Protocol ───────────────────────────────────────────────────────
    ACTIVE_CKPT = os.path.join(os.getcwd(), "checkpoints", "active_search_checkpoint.pth")
    os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "pths"), exist_ok=True)

    resume_state    = None
    hi_resume_state = None
    iteration       = 0
    results_log     = []
    best_config     = best_d_config = best_acc_config = best_acc_d_config = None
    best_acc        = best_acc_val = 0.0
    evaluated_lo    = False
    skip_hi_val     = args.skip_hi_val

    if os.path.exists(ACTIVE_CKPT):
        print("=" * 65)
        print("ACTIVE CHECKPOINT DETECTED — Resurrecting search state...")
        resume_state = torch.load(ACTIVE_CKPT, map_location=device)
        ss = resume_state.get('search_state', {})

        # Guardrail: discard checkpoint if search bounds changed
        if 'initial_lo' in ss:
            if (ss['initial_lo'] != args.lo or ss['initial_hi'] != args.hi or
                    ss['initial_lo_d'] != args.lo_depth or ss['initial_hi_d'] != args.hi_depth):
                print("  Checkpoint belongs to a different search space — discarding.")
                os.remove(ACTIVE_CKPT)
                resume_state = None
                ss = {}

        if ss and resume_state is not None:
            if ss.get('phase') == 'hi_validation':
                hi_resume_state     = resume_state
                _hi_resume_epoch    = hi_resume_state.get('epoch', 0) + 1
                resume_state        = None
                print(f"  Resuming HI-VALIDATION from epoch {_hi_resume_epoch}.")
            elif 'lo' in ss:
                args.lo, args.hi         = ss['lo'], ss['hi']
                args.lo_depth, args.hi_depth = ss['lo_d'], ss['hi_d']
                iteration                = ss['iteration']
                best_config              = ss['best_config']
                best_d_config            = ss['best_d_config']
                best_acc                 = ss['best_acc']
                best_acc_config          = ss['best_acc_config']
                best_acc_d_config        = ss['best_acc_d_config']
                best_acc_val             = ss['best_acc_val']
                results_log              = ss['results_log']
                evaluated_lo             = ss['evaluated_lo']
                skip_hi_val              = ss.get('skip_hi_val', skip_hi_val)
                print(f"  Search state revived — continuing from iteration {iteration}.")
        print("=" * 65 + "\n")

    # ── Load Teacher ──────────────────────────────────────────────────────────
    def load_teacher_safe(path, device, retries=3):
        if not os.path.exists(path):
            raise FileNotFoundError(f"\nTeacher not found at '{path}'.")
        for attempt in range(retries):
            try:
                print(f"Loading Teacher from '{path}' (Attempt {attempt+1}/{retries})...")
                t = models.resnet50(weights=None)
                t.fc = nn.Linear(t.fc.in_features, 10)
                t.load_state_dict(torch.load(path, map_location=device))
                t = t.to(device).eval()
                print(f"Teacher loaded.\n")
                return t
            except Exception:
                time.sleep(5)
        raise RuntimeError("Failed to load teacher after multiple attempts.")

    teacher = load_teacher_safe(args.teacher, device)
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs — DataParallel for Teacher.")
        teacher = nn.DataParallel(teacher)

    # ── Data Loaders (includes one-time logit cache) ──────────────────────────
    train_ld, val_ld = get_loaders(args.data, teacher=teacher, device=device, batch_size=args.batch)

    # ── Validate Teacher ──────────────────────────────────────────────────────
    print("Validating teacher accuracy...")
    correct = total = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            correct += (teacher(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    teacher_acc = correct / total
    print(f"Teacher Validation Accuracy: {teacher_acc:.4f}\n")
    if teacher_acc < args.teacher_min_acc:
        raise ValueError(f"Teacher only reached {teacher_acc*100:.2f}%. Aborting.")

    # ── Pre-validate hi (unless skipped) ─────────────────────────────────────
    if not skip_hi_val and resume_state is None:
        print("=" * 65)
        print(f"Pre-validating hi_w={args.hi} hi_d={args.hi_depth}...")
        _hi_student = DynamicNet(args.hi, args.hi_depth).to(device)
        if torch.cuda.device_count() > 1:
            _hi_student = nn.DataParallel(_hi_student)

        w_str_hi    = "-".join(map(str, args.hi))
        d_str_hi    = "-".join(map(str, args.hi_depth))
        hi_ckpt_name = f"pths/student_w{w_str_hi}_d{d_str_hi}.pth"
        hi_search_state = {
            'phase':       'hi_validation',
            'initial_lo':  args.lo,    'initial_hi':  args.hi,
            'initial_lo_d': args.lo_depth, 'initial_hi_d': args.hi_depth,
        }

        _, _hi_curve = train_student(
            _hi_student, teacher, train_ld, val_ld,
            epochs=args.full_epochs, device=device, lr=args.lr, verbose=True,
            ckpt_path=hi_ckpt_name,
            active_ckpt_path=ACTIVE_CKPT,
            cfg=args.hi, cfg_d=args.hi_depth,
            search_state=hi_search_state,
            resume_state=hi_resume_state,
        )
        _hi_acc = _hi_curve[-1]

        if os.path.exists(hi_ckpt_name):
            os.rename(hi_ckpt_name, f"pths/student_w{w_str_hi}_d{d_str_hi}_acc{_hi_acc:.4f}.pth")
        if os.path.exists(ACTIVE_CKPT):
            os.remove(ACTIVE_CKPT)

        print(f"  hi model final acc: {_hi_acc:.4f}\n")
        if _hi_acc < args.target_acc:
            raise ValueError(
                f"ABORT: hi model reached only {_hi_acc*100:.2f}% "
                f"(target: {args.target_acc*100:.0f}%)."
            )
        del _hi_student

    elif skip_hi_val:
        print(f"[STRATEGY] Skipping hi-validation (--skip-hi-val).\n")

    # ── 2D Binary Search ──────────────────────────────────────────────────────
    lo,   hi   = list(args.lo),    list(args.hi)
    lo_d, hi_d = list(args.lo_depth), list(args.hi_depth)

    def check_convergence(l_w, h_w, l_d, h_d):
        """Stop when the MB gap between lo and hi is negligible."""
        lo_mb_val = size_mb(DynamicNet(l_w, l_d), dtype_bytes=2)
        hi_mb_val = size_mb(DynamicNet(h_w, h_d), dtype_bytes=2)
        if (hi_mb_val - lo_mb_val) < args.size_tol_mb:
            return True
        # Fallback: also stop if channel-level tol is satisfied
        return configs_converged(l_w, h_w, tol=args.tol) and configs_converged(l_d, h_d, tol=1)

    print("=" * 65)
    print("Starting 2D Binary Search")
    print(f"  Variable epochs: base={args.full_epochs} max={args.max_epochs} scale={args.epoch_scale}")
    print(f"  Convergence:     size-gap < {args.size_tol_mb} MB")
    print("=" * 65)

    while not check_convergence(lo, hi, lo_d, hi_d):

        # ── Pick next config ──────────────────────────────────────────────────
        if resume_state is not None:
            cfg, cfg_d = resume_state['cfg'], resume_state['cfg_d']
        else:
            iteration += 1
            if not evaluated_lo:
                cfg, cfg_d   = list(lo), list(lo_d)
                evaluated_lo = True
                print("\n  Evaluating pure 'lo' bound first...")
            else:
                cfg   = midpoint(lo, hi, step=args.step)
                cfg_d = midpoint_depth(lo_d, hi_d)

        mb     = size_mb(DynamicNet(cfg, cfg_d), dtype_bytes=2)
        params = param_count(DynamicNet(cfg, cfg_d))

        # ── Variable epoch budget ─────────────────────────────────────────────
        node_epochs = min(
            int(args.full_epochs * (HI_MB / max(mb, 1e-9)) ** args.epoch_scale),
            args.max_epochs
        )
        proxy_ep = node_epochs // 2

        print(f"\n[Iter {iteration}]  w={cfg} d={cfg_d}  "
              f"({params:,} params  {mb:.4f} MB FP16)")
        print(f"  lo_w={lo}  hi_w={hi}  |  lo_d={lo_d}  hi_d={hi_d}")
        print(f"  Budget: {node_epochs} epochs  (proxy @ {proxy_ep})")
        print("─" * 65)

        student = DynamicNet(cfg, cfg_d).to(device)
        if torch.cuda.device_count() > 1:
            print(f"  DataParallel for Student [w={cfg}, d={cfg_d}]")
            student = nn.DataParallel(student)

        search_state = {
            'initial_lo':    args.lo,    'initial_hi':    args.hi,
            'initial_lo_d':  args.lo_depth, 'initial_hi_d': args.hi_depth,
            'lo':            list(lo),   'hi':            list(hi),
            'lo_d':          list(lo_d), 'hi_d':          list(hi_d),
            'iteration':     iteration,
            'best_config':   best_config,    'best_d_config':   best_d_config,
            'best_acc':      best_acc,
            'best_acc_config':  best_acc_config,
            'best_acc_d_config': best_acc_d_config,
            'best_acc_val':  best_acc_val,
            'results_log':   results_log,
            'evaluated_lo':  evaluated_lo,
            'skip_hi_val':   skip_hi_val,
        }

        w_str, d_str = "-".join(map(str, cfg)), "-".join(map(str, cfg_d))
        ckpt_name = f"pths/student_w{w_str}_d{d_str}_acc"

        tmp_resume = resume_state
        resume_state = None

        _, acc_curve = train_student(
            student, teacher, train_ld, val_ld,
            epochs=node_epochs,
            device=device,
            lr=args.lr,
            verbose=True,
            proxy_epochs=proxy_ep,
            total_epochs=node_epochs,
            target_acc=args.target_acc,
            active_ckpt_path=ACTIVE_CKPT,
            cfg=cfg, cfg_d=cfg_d,
            search_state=search_state,
            resume_state=tmp_resume,
        )

        log_entry = {
            "iteration":   iteration,
            "config_w":    cfg,
            "config_d":    cfg_d,
            "mb_fp16":     round(mb, 4),
            "node_epochs": node_epochs,
            "proxy_ep":    proxy_ep,
            "epochs_run":  len(acc_curve),
            "proxy_acc":   acc_curve[proxy_ep - 1] if len(acc_curve) >= proxy_ep and proxy_ep > 0 else None,
            "full_acc":    None,
            "verdict":     None,
        }

        # ── Verdict ───────────────────────────────────────────────────────────
        raw_state = (student.module.state_dict()
                     if isinstance(student, nn.DataParallel)
                     else student.state_dict())

        if len(acc_curve) <= proxy_ep:
            # Trajectory projection cut this model early
            log_entry["verdict"] = "trajectory_cut"
            print(f"  Cut at proxy epoch {len(acc_curve)} — INSUFFICIENT → lo = current")
            if cfg == lo and cfg_d == lo_d:
                lo   = [l + args.step if l < h else l for l, h in zip(lo, hi)]
                lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
            else:
                lo, lo_d = list(cfg), list(cfg_d)

        else:
            full_acc = acc_curve[-1]
            log_entry["full_acc"] = full_acc

            # Save FP16 weights for every fully-trained node
            torch.save({k: v.cpu().clone().half() for k, v in raw_state.items()},
                       f"{ckpt_name}{full_acc:.4f}.pth")

            if full_acc >= args.target_acc:
                print(f"  SUFFICIENT ({full_acc:.4f} >= {args.target_acc}) → hi = current")
                log_entry["verdict"] = "sufficient"
                if cfg == hi and cfg_d == hi_d:
                    hi   = [h - args.step if h > l else h for l, h in zip(lo, hi)]
                    hi_d = [hd - 1 if hd > ld else hd for ld, hd in zip(lo_d, hi_d)]
                else:
                    hi, hi_d = list(cfg), list(cfg_d)

                # Track smallest passing model
                if (best_config is None or
                        size_mb(DynamicNet(cfg, cfg_d), dtype_bytes=2) <
                        size_mb(DynamicNet(best_config, best_d_config), dtype_bytes=2)):
                    best_config, best_d_config, best_acc = cfg, cfg_d, full_acc
                    _tmp = DynamicNet(cfg, cfg_d)
                    _tmp.load_state_dict(raw_state)
                    torch.save(_tmp.half().state_dict(), args.out)
                    print(f"  New size record! Saved FP16 to '{args.out}'")

            else:
                print(f"  INSUFFICIENT ({full_acc:.4f} < {args.target_acc}) → lo = current")
                log_entry["verdict"] = "insufficient"
                if cfg == lo and cfg_d == lo_d:
                    lo   = [l + args.step if l < h else l for l, h in zip(lo, hi)]
                    lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
                else:
                    lo, lo_d = list(cfg), list(cfg_d)

            # Track highest accuracy seen (regardless of size)
            if full_acc > best_acc_val:
                best_acc_val, best_acc_config, best_acc_d_config = full_acc, cfg, cfg_d
                _tmp = DynamicNet(cfg, cfg_d)
                _tmp.load_state_dict(raw_state)
                torch.save(_tmp.half().state_dict(), args.out + ".best_acc.pth")
                print(f"  New accuracy record ({full_acc:.4f})! Saved to '{args.out}.best_acc.pth'")

            if best_config:
                best_mb = size_mb(DynamicNet(best_config, best_d_config), dtype_bytes=2)
                print(f"  Best-by-SIZE: w={best_config} d={best_d_config} "
                      f"| {best_mb:.4f} MB | acc={best_acc:.4f}")

        results_log.append(log_entry)
        with open("search_results.json", "w") as f:
            json.dump(results_log, f, indent=2)

        # Clean up checkpoint for this node (next node will write its own)
        if os.path.exists(ACTIVE_CKPT):
            os.remove(ACTIVE_CKPT)

    # ── Final Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Search Converged!")
    print(f"  lo: w={lo} d={lo_d}  ({size_mb(DynamicNet(lo, lo_d), 2):.4f} MB)")
    print(f"  hi: w={hi} d={hi_d}  ({size_mb(DynamicNet(hi, hi_d), 2):.4f} MB)")
    if best_config:
        best_mb = size_mb(DynamicNet(best_config, best_d_config), dtype_bytes=2)
        print(f"  Best-by-SIZE: w={best_config} d={best_d_config} "
              f"| {best_mb:.4f} MB | acc={best_acc:.4f}")
    print("=" * 65)

    with open("search_results.json", "w") as f:
        json.dump(results_log, f, indent=2)


if __name__ == '__main__':
    main()
