"""Benchmark every saved pair result by running both real models sequentially."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from common.latency import forward_latency
from dino.baseline.model import WireModel as DinoModel
from ensemble.stack_pairs import parse_model, write_comparison, write_summary
from mobilenet_v3_small.model import WireModel as MobileNetModel
from repvit_m0_9.model import WireModel as RepViTModel
from resnet.model import WireModel as ResNetModel


class RepViTAdapter(torch.nn.Module):
    """Expose RepViT's dictionary input through the shared tensor interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, history):
        return self.model({
            "goal": images[:, 0],
            "frames": images[:, 1:],
            "depth_relative_mm": history[:, :6],
            "history_offsets_s": history[:, 6:],
        })


def load_model(name, checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("config", {})
    if name == "resnet":
        model = ResNetModel(pretrained=False, dropout=config.get("dropout", 0.3),
                            finetune_layer4=config.get("finetune_layer4", False))
    elif name == "mobilenet":
        model = MobileNetModel(pretrained=False)
    elif name == "dino":
        model = DinoModel(saved["image_size"], pretrained=False,
                          dropout=config.get("dropout", 0.3))
    elif name == "repvit":
        model = RepViTModel(
            pretrained=False,
            freeze_backbone=not config.get("finetune", False),
            hidden_dim=config.get("protocol_settings", {}).get("hidden_dim", 128),
            dropout=config.get("dropout", 0.3),
            finetune_last_stage=config.get("finetune_last_stage", False),
            history_order=config.get("history_order", "depth_then_time"),
        )
    else:
        raise ValueError(f"unsupported latency model: {name}")
    model.load_state_dict(saved.get("weights", saved.get("model")))
    image_size = saved.get("image_size", config.get("image_size"))
    if name == "repvit":
        model = RepViTAdapter(model)
    return model.to(device).eval(), image_size


def stacker_forward(first, second, images, history, parameters, threshold):
    first_probability = first(images, history).softmax(1)[:, 1].clamp(1e-3, 1 - 1e-3)
    second_probability = second(images, history).softmax(1)[:, 1].clamp(1e-3, 1 - 1e-3)
    x = torch.stack((torch.logit(first_probability), torch.logit(second_probability)), dim=1)
    normalized = (x - parameters["mean"]) / parameters["std"]
    probability = torch.sigmoid(
        (normalized * parameters["weight"]).sum(1) + parameters["bias"])
    return probability >= threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="Repeat NAME=ARTIFACT_ROOT")
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run in a GPU allocation")
    models = dict(args.model)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = torch.cuda.get_device_name()

    results = {}
    for result_path in sorted(args.results_root.glob("*__*/results.json")):
        result = json.loads(result_path.read_text())
        first_name, second_name = result["models"]
        if first_name not in models or second_name not in models:
            continue
        if "fold_stackers" not in result:
            raise ValueError(f"{result_path} has no fitted fold stackers; evaluate test first")
        samples = []
        fold_latency = []
        for fold in range(5):
            first_checkpoint = models[first_name] / f"fold{fold}" / "best.pt"
            second_checkpoint = models[second_name] / f"fold{fold}" / "best.pt"
            first, first_size = load_model(first_name, first_checkpoint, device)
            second, second_size = load_model(second_name, second_checkpoint, device)
            if first_size != second_size:
                raise ValueError("latency benchmark currently requires matching image sizes")
            images = torch.zeros(1, 7, 3, first_size, first_size, device=device)
            history = torch.zeros(1, 12, device=device)
            saved_parameters = result["fold_stackers"][fold]
            parameters = {
                "weight": torch.tensor(saved_parameters["standardized_weights"],
                                       device=device),
                "bias": torch.tensor(saved_parameters["bias"], device=device),
                "mean": torch.tensor(saved_parameters["feature_mean"], device=device),
                "std": torch.tensor(saved_parameters["feature_std"], device=device),
            }
            latency = forward_latency(
                lambda: stacker_forward(first, second, images, history, parameters,
                                         result["decision_threshold"]))
            samples.extend(latency["samples_ms"])
            fold_latency.append({"fold": fold, "mean_ms": latency["mean_ms"],
                                 "median_ms": latency["median_ms"],
                                 "warmup": latency["warmup"],
                                 "repeats": latency["repeats"]})
            del first, second, images, history
            torch.cuda.empty_cache()
        result.update({
            "ensemble_forward_latency_mean_ms": statistics.mean(samples),
            "ensemble_forward_latency_median_ms": statistics.median(samples),
            "latency_measurements": len(samples),
            "latency_folds": fold_latency,
            "latency_scope": "sequential batch-one forwards of both models plus logistic stacker; GPU-resident inputs; excludes I/O, preprocessing and transfer",
            "latency_precision": "FP32; TF32 disabled",
            "gpu": gpu,
            "latency_status": "complete",
        })
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        write_summary(result_path.with_name("summary.csv"), result)
        results[result_path.parent.name] = result
        print(result_path.parent.name, result["ensemble_forward_latency_median_ms"])

    if not results:
        parser.error("no compatible pair result directories found")
    write_comparison(args.results_root / "comparison.csv", results)


if __name__ == "__main__":
    main()
