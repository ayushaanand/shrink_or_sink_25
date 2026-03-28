"""
search.py
---------
Interpolating Binary Search over DynamicNet configurations, now supporting 2D search (Width & Depth).
Features native 100% Kaggle-Timeout Rescue Checkpointing.
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
parser.add_argument("--full-epochs",   type=int,   default=40)
parser.add_argument("--proxy-thresh",  type=float, default=0.65,
                    help="Min val acc at proxy-epochs to proceed to full training")
parser.add_argument("--target-acc",    type=float, default=0.85)
parser.add_argument("--batch",         type=int,   default=128)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--teacher-min-acc", type=float, default=0.82)
parser.add_argument("--proxy-ratio",   type=float, default=0.85)
parser.add_argument("--direct-thresh", type=float, default=None)
parser.add_argument("--out",           type=str,   default="best_student.pth")
args = parser.parse_args()


def main():
    assert len(args.lo) == len(args.hi) == len(args.lo_depth) == len(args.hi_depth), \
        f"All bound configs (--lo, --hi, --lo-depth, --hi-depth) must have the exact same number of elements."

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Validate that hi config itself fits under 5 MB ────────────────────────────
    hi_model = DynamicNet(args.hi, args.hi_depth)
    assert size_mb(hi_model, dtype_bytes=2) < 5.0, (
        f"--hi config is already {size_mb(hi_model, 2):.3f} MB (FP16) — exceeds 5 MB limit!"
    )
    print(f"Search bounds:")
    print(f"  lo = w:{args.lo} d:{args.lo_depth}  ({size_mb(DynamicNet(args.lo, args.lo_depth), 2):.4f} MB FP16)")
    print(f"  hi = w:{args.hi} d:{args.hi_depth}  ({size_mb(DynamicNet(args.hi, args.hi_depth), 2):.4f} MB FP16)\n")

    # ── Resume Protocol / Preamble Check ──────────────────────────────────────────
    ACTIVE_CKPT = "checkpoints/active_search_checkpoint.pth"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("pths", exist_ok=True)

    resume_state = None
    iteration = 0
    results_log = []
    best_config = best_d_config = best_acc_config = best_acc_d_config = None
    best_acc = best_acc_val = 0.0
    evaluated_lo = False
    dynamic_proxy_thresh = None

    if os.path.exists(ACTIVE_CKPT):
        print("=" * 65)
        print("🚨 ACTIVE CHECKPOINT DETECTED! Resurrecting global search state...")
        resume_state = torch.load(ACTIVE_CKPT, map_location=device)
        ss = resume_state.get('search_state', {})
        
        # ── Verify Search Config Guardrail ─────────────────────────────────────────
        if 'initial_lo' in ss:
            if ss['initial_lo'] != args.lo or ss['initial_hi'] != args.hi or \
               ss['initial_lo_d'] != args.lo_depth or ss['initial_hi_d'] != args.hi_depth:
                print("🚨 WARNING: Active checkpoint belongs to a completely different (--lo, --hi) search space!")
                print("   Discarding the old checkpoint and starting fresh.")
                os.remove(ACTIVE_CKPT)
                resume_state = None
                ss = {}

        if 'lo' in ss and resume_state is not None:
            args.lo, args.hi = ss['lo'], ss['hi']
            args.lo_depth, args.hi_depth = ss['lo_d'], ss['hi_d']
            iteration = ss['iteration']
            best_config, best_d_config, best_acc = ss['best_config'], ss['best_d_config'], ss['best_acc']
            best_acc_config, best_acc_d_config = ss['best_acc_config'], ss['best_acc_d_config']
            best_acc_val = ss['best_acc_val']
            results_log = ss['results_log']
            evaluated_lo = ss['evaluated_lo']
            dynamic_proxy_thresh = ss['dynamic_proxy_thresh']
            print(f"  [STRATEGY] Search State completely revived! Jumping cleanly into iteration {iteration}.")
        print("=" * 65 + "\n")

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
                time.sleep(5)
        raise RuntimeError("🚨 CRITICAL ERROR: Failed to load teacher after multiple attempts.")

    teacher = load_teacher_safe(args.teacher, device)
    if torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs! Enabling DataParallel for Teacher.")
        teacher = nn.DataParallel(teacher)

    # ── Data loaders ──────────────────────────────────────────────────────────────
    train_ld, val_ld = get_loaders(args.data, teacher=teacher, device=device, batch_size=args.batch)

    print("Evaluating loaded teacher model on validation set to ensure quality...")
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            out = teacher(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    teacher_acc = correct / total
    print(f"Teacher Validation Accuracy: {teacher_acc:.4f}\n")

    if teacher_acc < args.teacher_min_acc:
        raise ValueError(f"🚨 CRITICAL ERROR: Teacher only reached {teacher_acc*100:.2f}%. Aborting.")

    # ── Pre-validate hi: calibrate proxy threshold ─────────────────────────────────
    if resume_state is None:
        if args.direct_thresh is not None:
            dynamic_proxy_thresh = args.direct_thresh
            print(f"[STRATEGY] Hardcoding dynamic_proxy_thresh = {dynamic_proxy_thresh:.4f}\n")
        else:
            print("=" * 65)
            print(f"Pre-validating upper bound hi_w={args.hi} hi_d={args.hi_depth}...")
            _hi_student = DynamicNet(args.hi, args.hi_depth).to(device)
            if torch.cuda.device_count() > 1:
                _hi_student = nn.DataParallel(_hi_student)

            w_str_hi, d_str_hi = "-".join(map(str, args.hi)), "-".join(map(str, args.hi_depth))
            hi_ckpt_name = f"pths/student_w{w_str_hi}_d{d_str_hi}.pth"

            _, _hi_curve = train_student(
                _hi_student, teacher, train_ld, val_ld,
                epochs=args.full_epochs, device=device, lr=args.lr, verbose=True,
                ckpt_path=hi_ckpt_name
            )
            hi_proxy_acc = _hi_curve[args.proxy_epochs - 1]
            _hi_acc = _hi_curve[-1]
            
            if os.path.exists(hi_ckpt_name):
                os.rename(hi_ckpt_name, f"pths/student_w{w_str_hi}_d{d_str_hi}_acc{_hi_acc:.4f}.pth")
            
            dynamic_proxy_thresh = round(hi_proxy_acc * args.proxy_ratio, 4)
            print(f"  → dynamic_proxy_thresh = {dynamic_proxy_thresh:.4f}\n")
            
            if _hi_acc < args.target_acc:
                raise ValueError(f"🚨 ABORT: Upper bound hi only reached {_hi_acc*100:.2f}% (Target: {args.target_acc*100:.0f}%).")
            del _hi_student

    # ── Interpolating Binary Search ───────────────────────────────────────────────
    lo, hi = list(args.lo), list(args.hi)
    lo_d, hi_d = list(args.lo_depth), list(args.hi_depth)

    print("=" * 65)
    print("Starting 2D Interpolating Binary Search")
    print("=" * 65)

    def check_convergence(l_w, h_w, l_d, h_d):
        return configs_converged(l_w, h_w, tol=args.tol) and configs_converged(l_d, h_d, tol=1)

    while not check_convergence(lo, hi, lo_d, hi_d):
        if resume_state is not None:
            cfg, cfg_d = resume_state['cfg'], resume_state['cfg_d']
        else:
            iteration += 1
            if not evaluated_lo:
                cfg, cfg_d = list(lo), list(lo_d)
                evaluated_lo = True
                print("\n  [STRATEGY] Evaluating pure 'lo' bound first...")
            else:
                cfg = midpoint(lo, hi, step=args.step)
                cfg_d = midpoint_depth(lo_d, hi_d)

        mb = size_mb(DynamicNet(cfg, cfg_d), dtype_bytes=2)
        params = param_count(DynamicNet(cfg, cfg_d))
        
        print(f"\n[Iter {iteration}]  Probing w={cfg} d={cfg_d}  ({params:,} params  {mb:.4f} MB FP16)")
        print(f"  lo_w={lo}  hi_w={hi}    lo_d={lo_d}  hi_d={hi_d}")
        print("─" * 65)

        student = DynamicNet(cfg, cfg_d).to(device)
        if torch.cuda.device_count() > 1:
            print(f"🚀 Enabling DataParallel for Student [w={cfg}, d={cfg_d}]")
            student = nn.DataParallel(student)

        # Pack exact state for robust recovery
        search_state = {
            'initial_lo': args.lo, 'initial_hi': args.hi,
            'initial_lo_d': args.lo_depth, 'initial_hi_d': args.hi_depth,
            'lo': list(lo), 'hi': list(hi), 'lo_d': list(lo_d), 'hi_d': list(hi_d),
            'iteration': iteration,
            'best_config': best_config, 'best_d_config': best_d_config, 'best_acc': best_acc,
            'best_acc_config': best_acc_config, 'best_acc_d_config': best_acc_d_config, 'best_acc_val': best_acc_val,
            'results_log': results_log,
            'evaluated_lo': evaluated_lo,
            'dynamic_proxy_thresh': dynamic_proxy_thresh
        }

        w_str, d_str = "-".join(map(str, cfg)), "-".join(map(str, cfg_d))
        ckpt_name = f"pths/student_w{w_str}_d{d_str}_acc"

        tmp_resume = resume_state
        resume_state = None

        _, acc_curve = train_student(
            student, teacher, train_ld, val_ld,
            epochs=args.full_epochs, device=device, lr=args.lr, verbose=True,
            proxy_epochs=args.proxy_epochs, proxy_thresh=dynamic_proxy_thresh,
            active_ckpt_path=ACTIVE_CKPT,
            cfg=cfg, cfg_d=cfg_d,
            search_state=search_state,
            resume_state=tmp_resume
        )

        log_entry = {
            "iteration": iteration,
            "config_w":  cfg,
            "config_d":  cfg_d,
            "mb_fp16":   round(mb, 4),
            "proxy_acc": None,
            "full_acc":  None,
            "verdict":   None,
        }

        if len(acc_curve) <= args.proxy_epochs:
            proxy_acc = acc_curve[-1]
            log_entry["proxy_acc"] = proxy_acc
            print(f"  ✗ Proxy {proxy_acc:.4f} < {dynamic_proxy_thresh:.4f} → TOO SMALL → lo = current\n")
            log_entry["verdict"] = "too_small_proxy"
            if cfg == lo and cfg_d == lo_d:
                lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
                lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
            else:
                lo, lo_d = cfg, cfg_d
        else:
            log_entry["proxy_acc"] = acc_curve[args.proxy_epochs - 1]
            full_acc = acc_curve[-1]
            log_entry["full_acc"] = full_acc
            print(f"  Full best acc:  {full_acc:.4f}")
            
            # Save microscopic FP16
            raw_state = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            torch.save({k: v.cpu().clone().half() for k, v in raw_state.items()}, f"{ckpt_name}{full_acc:.4f}.pth")

            if full_acc >= args.target_acc:
                print(f"  ✓ {full_acc:.4f} ≥ {args.target_acc} — SUFFICIENT → hi = current")
                log_entry["verdict"] = "sufficient"
                if cfg == hi and cfg_d == hi_d:
                    hi = [h - args.step if h > l else h for l, h in zip(lo, hi)]
                    hi_d = [hd - 1 if hd > ld else hd for ld, hd in zip(lo_d, hi_d)]
                else:
                    hi, hi_d = cfg, cfg_d
                    
                if best_config is None or size_mb(DynamicNet(cfg, cfg_d)) < size_mb(DynamicNet(best_config, best_d_config)):
                    best_config, best_d_config, best_acc = cfg, cfg_d, full_acc
                    _tmp = DynamicNet(cfg, cfg_d)
                    _tmp.load_state_dict(raw_state)
                    torch.save(_tmp.half().state_dict(), args.out)
                    print(f"  🏆 New Size record! Saved FP16 to '{args.out}'")
            else:
                print(f"  ✗ {full_acc:.4f} < {args.target_acc} — INSUFFICIENT → lo = current")
                log_entry["verdict"] = "insufficient"
                if cfg == lo and cfg_d == lo_d:
                    lo = [l + args.step if l < h else l for l, h in zip(lo, hi)]
                    lo_d = [ld + 1 if ld < hd else ld for ld, hd in zip(lo_d, hi_d)]
                else:
                    lo, lo_d = cfg, cfg_d

            if full_acc > best_acc_val:
                best_acc_val, best_acc_config, best_acc_d_config = full_acc, cfg, cfg_d
                _tmp = DynamicNet(cfg, cfg_d)
                _tmp.load_state_dict(raw_state)
                torch.save(_tmp.half().state_dict(), args.out + ".best_acc.pth")
                print(f"  🥇 New accuracy record! saved to '{args.out}.best_acc.pth'")

            if best_config:
                print(f"     Best-by-SIZE : w={best_config} d={best_d_config} | acc={best_acc:.4f}")

        results_log.append(log_entry)
        
        if os.path.exists(ACTIVE_CKPT):
            os.remove(ACTIVE_CKPT)
            print(f"  [CLEANUP] Safely nuked active checkpoint heavily tracking memory.")


    # ── Final Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Search Converged!")
    print(f"  Final bounds: {lo} -> {hi} | {lo_d} -> {hi_d}")
    print("=" * 65)

    with open("search_results.json", "w") as f:
        json.dump(results_log, f, indent=2)


if __name__ == '__main__':
    main()
