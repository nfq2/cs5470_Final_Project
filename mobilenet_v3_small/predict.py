"""Predict one supplied case on Perlmutter; outputs only the action."""
import argparse
import json
from pathlib import Path

import torch

from data import WireDataset
from model import WireModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with (Path(args.data_root).expanduser() / "cases.jsonl").open() as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if not 0 <= args.case_index < len(cases):
        parser.error("case-index is outside the supplied case list")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = WireModel().to(args.device)
    model.load_state_dict(saved["weights"])
    model.eval()
    images, history, _ = WireDataset(
        args.data_root, cases, [args.case_index], saved["image_size"]
    )[0]
    with torch.no_grad():
        label = model(images.unsqueeze(0).to(args.device),
                      history.unsqueeze(0).to(args.device)).argmax(1).item()
    print("ADVANCE" if label else "RETRACT")
