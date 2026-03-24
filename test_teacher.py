import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import sys

def test_teacher(model_path, data_path="./data"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    ])

    print("Loading test set...")
    test_ds = datasets.STL10(root=data_path, split="test", download=True, transform=transform)
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print("Loading ResNet-50 Teacher...")
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    
    ckpt = torch.load(model_path, map_location=device)
    teacher.load_state_dict(ckpt)
    
    # Optional DataParallel if running on a Multi-GPU machine
    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)
        
    teacher = teacher.to(device)
    teacher.eval()

    correct = 0
    total = len(test_ds)

    print("Running inference...")
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            preds = teacher(x).argmax(dim=1)
            correct += (preds == y).sum().item()

    accuracy = (correct / total) * 100.0
    print("-" * 40)
    print(f"Teacher Final Test Accuracy: {accuracy:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_teacher.py <model_path>")
    else:
        test_teacher(sys.argv[1])
