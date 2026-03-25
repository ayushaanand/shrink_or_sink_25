import torch
try:
    if torch.cuda.is_available():
        mask = torch.tensor([True, False], device='cuda')
        indices = torch.tensor([0, 1])
        res = indices[mask]
        print("Success!")
    else:
        print("CUDA not available, but error usually happens if devices mismatch.")
except Exception as e:
    print("Error:", e)
