import argparse, re, torch, torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import DynamicNet

def infer_architecture(state_dict):
    new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if "layer1.0.conv1.weight" in new_sd:
        return None, None, new_sd, "teacher"
    if "conv1.0.weight" in new_sd:
        stage_out, stage_max = {}, {}
        pat = re.compile(r"^features\.(\d+)\.stage\.(\d+)\.pw_bn\.weight$")
        for k, v in new_sd.items():
            m = pat.match(k)
            if m:
                s, d = int(m.group(1)), int(m.group(2))
                stage_out[s] = v.shape[0]
                stage_max[s] = max(stage_max.get(s, 0), d)
        n = max(stage_out.keys()) + 1
        return [stage_out[s] for s in range(n)], [stage_max[s]+1 for s in range(n)], new_sd, "student"
    raise KeyError("Unknown Architecture")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--widths", type=int, nargs='+')
    p.add_argument("--depths", type=int, nargs='+')
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(args.model_path, map_location="cpu")
    if isinstance(ckpt, list):
        if isinstance(ckpt[0], list) and isinstance(ckpt[1], list):
            w, d, weights = ckpt[0], ckpt[1], ckpt[2:]
        else:
            w, d, weights = args.widths, args.depths, ckpt
        model = DynamicNet(widths=w, depths=d)
        keys = sorted(model.state_dict().keys())
        model.load_state_dict(dict(zip(keys, weights)))
    else:
        raw = ckpt.get("model", ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))))
        w_inf, d_inf, sd, m_type = infer_architecture(raw)
        w, d = (args.widths or w_inf), (args.depths or d_inf)
        if m_type == "teacher":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
        else:
            model = DynamicNet(widths=w, depths=d)
        model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.4467,0.4398,0.4066], std=[0.2603,0.2566,0.2713])])
    test_ds = torchvision.datasets.STL10(root=args.dataset_path, split="test", download=not args.no_download, transform=test_tf)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"\nAccuracy: {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
