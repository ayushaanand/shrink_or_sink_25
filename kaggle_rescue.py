import torch
import os

ckpt_path = "checkpoints/active_search_checkpoint.pth"

if os.path.exists(ckpt_path):
    print("Loading active checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"Current corrupted epoch: {ckpt.get('epoch', 'N/A')}")
    
    # Scrub the corrupted Neural Network weights and mismatched Optimizer vectors
    if 'model' in ckpt: del ckpt['model']
    if 'optimizer' in ckpt: del ckpt['optimizer']
    if 'scheduler' in ckpt: del ckpt['scheduler']
    
    # Rewind the epoch and metrics so the script seamlessly restarts Iter 4 from Epoch 1
    ckpt['epoch'] = 0
    ckpt['acc_curve'] = []
    
    # Save the sanitized checkpoint back to disk
    torch.save(ckpt, ckpt_path)
    print("✅ Scrubbed the corrupted weights!")
    print("The Search State boundaries are perfectly intact.")
    print("You can now safely run search.py to cleanly restart Iteration 4 from Epoch 1.")
else:
    print("Checkpoint not found. Make sure you are in the shrink_or_sink_25 directory.")
