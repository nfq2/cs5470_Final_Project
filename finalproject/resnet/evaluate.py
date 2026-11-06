"""python evaluate.py validation   (or: python evaluate.py test)"""
import json
import sys

import torch
from torch.utils.data import DataLoader

from data import WireDataset, load_data
from model import WireModel
from common.latency import forward_latency  # data.py puts the repository root on sys.path
from common.metrics import report


if __name__ == "__main__":
    split_name = sys.argv[1] if len(sys.argv) > 1 else "validation"
    assert split_name in ["validation", "test"], "Choose validation or test."
    assert torch.cuda.is_available(), "Allocate a GPU before running evaluate.py."
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load("resnet18.pt", map_location="cpu", weights_only=True)
    model = WireModel().to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    cases, split = load_data(checkpoint["fold"])
    dataset = WireDataset(cases, split[split_name], checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    labels, probs = [], []
    with torch.no_grad():
        for images, history, y in loader:
            outputs = model(images.to(device), history.to(device))
            labels.extend(y.tolist())
            probs.extend(outputs.softmax(1)[:, 1].cpu().tolist())  # P(advance)
    # Shared report: overall, advance, retract and balanced accuracy, AUROC, always-advance.
    results = {"split": split_name, "fold": checkpoint["fold"], "epoch": checkpoint["epoch"],
               **report(labels, probs)}

    # Time one whole decision: seven images and their numeric history.
    # Inputs are already on the GPU; preprocessing and transfer are excluded.
    images, history, _ = dataset[0]
    images = images.unsqueeze(0).to(device)
    history = history.unsqueeze(0).to(device)
    latency = forward_latency(lambda: model(images, history))
    results["forward_latency_mean_ms"] = latency["mean_ms"]
    results["forward_latency_median_ms"] = latency["median_ms"]
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps(results, indent=2))
    with open(split_name + "_results.json", "w") as f:
        json.dump(results, f, indent=2)
