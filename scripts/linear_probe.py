"""Linear probe of the frozen backbone (no prompts): a check on the backbone weights and preprocessing.

Extracts the final class-token features of train/val/test images once, fits a linear classifier with
full-batch L-BFGS for several L2 strengths, and reports the validation and test accuracy of each.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from pib.data import build_loader
from pib.utils import load_config


@torch.no_grad()
def features(backbone, loader, device):
    xs, ys = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            xs.append(backbone(batch["images"].to(device)).float())
        ys.append(batch["labels"].to(device))
    return torch.cat(xs), torch.cat(ys)


def fit(x, y, classes, l2, steps=200):
    weight = torch.zeros(x.shape[1], classes, device=x.device, requires_grad=True)
    bias = torch.zeros(classes, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weight + bias, y) + l2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    import timm

    config = load_config(args.config, [f"data.manifest={args.manifest}", "data.cache_in_memory=false"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = timm.create_model(config["model"]["backbone"], pretrained=True, num_classes=0).to(device).eval()
    data = {split: features(backbone, build_loader(config, split), device) for split in ("train", "val", "test")}
    classes = int(max(int(y.max()) for _, y in data.values())) + 1
    mean, std = data["train"][0].mean(0), data["train"][0].std(0) + 1e-6
    standardized = {split: ((x - mean) / std, y) for split, (x, y) in data.items()}
    results = []
    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
        weight, bias = fit(*standardized["train"], classes, l2)
        accuracy = {split: 100 * float(((x @ weight + bias).argmax(1) == y).float().mean())
                    for split, (x, y) in standardized.items()}
        results.append({"l2": l2, **accuracy})
        print(json.dumps(results[-1]), flush=True)
    best = max(results, key=lambda row: row["val"])
    print(json.dumps({"selected_by_val": best, "feature_norm": float(data["train"][0].norm(dim=1).mean())}))


if __name__ == "__main__":
    main()
