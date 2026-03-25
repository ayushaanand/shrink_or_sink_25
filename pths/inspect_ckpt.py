import torch
import sys

def inspect_checkpoint(ckpt_path):
    try:
        print(f"\n[Loading] '{ckpt_path}'...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if isinstance(ckpt, dict):
            print("--- Checkpoint Information ---")
            
            # Print Epoch and Accuracy if available
            if 'epoch' in ckpt:
                print(f"Epoch: {ckpt['epoch']}")
            if 'best_acc' in ckpt:
                print(f"Best Validation Accuracy: {ckpt['best_acc'] * 100:.2f}%")
            
            # Inspect actual model state
            if 'model_state' in ckpt:
                m_state = ckpt['model_state']
                keys = list(m_state.keys())
                has_module = any(k.startswith('module.') for k in keys)
                print(f"Model keys total: {len(keys)}")
                print(f"Contains 'module.' prefix (DataParallel): {has_module}")
            else:
                keys = list(ckpt.keys())
                has_module = any(k.startswith('module.') for k in keys)
                print("No 'model_state' dictionary found. This file is pure weights.")
                print(f"Model keys total: {len(keys)}")
                print(f"Contains 'module.' prefix (DataParallel): {has_module}")
            
            # Check for training states
            if 'optimizer_state' in ckpt:
                print("Optimizer state included: Yes")
            if 'scheduler_state' in ckpt:
                print("Scheduler state included: Yes")
        
        else:
            print("Checkpoint is not a dictionary. It's probably just state_dict weights.")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "teacher_final.pth"
    inspect_checkpoint(path)
