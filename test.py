"""
test.py
-------
Implements the evaluation/inference pipeline for the Shrink or Sink competition.
Loads the final submission `.pth` file and evaluates on the STL-10 test set.

Usage:
    python test.py --dataset-path ./data --model-path final_model.pth
"""

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DynamicNet

def evaluate(dataset_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # STL-10 basic validation transforms (no augmentation!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])

    print(f"Loading test set from '{dataset_path}'...")
    test_ds = datasets.STL10(root=dataset_path, split="test", download=True, transform=transform)
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print("Initializing model skeleton from model.py...")
    model = DynamicNet() # Automatically uses WINNING_WIDTHS fallback
    
    print(f"Loading weights from '{model_path}'...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = len(test_ds)

    print("Running inference...")
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()

    accuracy = (correct / total) * 100.0
    print("-" * 40)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print("-" * 40)
    
    if accuracy >= 85.0:
        print("✅ Threshold passed! This model is eligible for score points.")
    else:
        print("❌ Model did not achieve 85% accuracy threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to STL-10 dataset folder")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .pth file")
    args = parser.parse_args()
    
    evaluate(args.dataset_path, args.model_path)
