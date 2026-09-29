import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .data import build_loader
from .evaluate import load_checkpoint_model
from .losses import compression_loss, layer_schedule, scatter_statistics, sufficiency_loss


def build_diagnostic_loader(config, split="test"):
    loader = build_loader(config, split)
    generator = torch.Generator().manual_seed(int(config.get("data", {}).get("diagnostic_seed", 42)))
    sampler = RandomSampler(loader.dataset, replacement=False, generator=generator)
    return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                      worker_init_fn=loader.worker_init_fn, generator=generator)


def diagnose(checkpoint, output, manifest=None, root=None, split="test", device=None):
    model, config, device = load_checkpoint_model(checkpoint, device)
    if manifest:
        config["data"]["manifest"] = manifest
    if root:
        config["data"]["root"] = root
    loader = build_diagnostic_loader(config, split)
    alpha = layer_schedule(config.get("loss", {}).get("alpha", 1.0), len(model.prompt_layers))
    summaries, count = {}, 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            prediction = model(batch["images"].to(device), return_states=True)
            states = prediction["states"]
            final_pooled = prediction["final_pooled"]
            batch_size = labels.numel()
            for index, state in enumerate(states):
                comp = compression_loss(state.tokens)
                suff = sufficiency_loss(state.pooled, labels)
                intra, inter, valid, classes = scatter_statistics(state.pooled, labels)
                values = {"compression": float(comp), "redundancy": float(comp) / max(1, state.tokens.shape[-1] * (state.tokens.shape[-1] - 1)),
                          "sufficiency": float(suff), "path_score": 1 / (float(comp + alpha[index] * suff) + 1e-8),
                          "separability": float(inter / (intra + 1e-8)) if valid and classes > 1 else 0.0}
                if state.pooled.shape[-1] == final_pooled.shape[-1]:
                    values["trajectory"] = float(F.cosine_similarity(state.pooled.float(), final_pooled.float()).mean())
                row = summaries.setdefault(state.index, {})
                for key, value in values.items():
                    row[key] = row.get(key, 0.0) + value * batch_size
            count += batch_size
    rows = [{"layer": layer, **{key: value / count for key, value in values.items()}}
            for layer, values in summaries.items()]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("layer", "compression", "redundancy", "sufficiency", "path_score", "separability", "trajectory")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export layer-wise PIB diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/diagnostics.csv")
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    args = parser.parse_args()
    diagnose(args.checkpoint, args.output, args.manifest, args.root, args.split, args.device)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
