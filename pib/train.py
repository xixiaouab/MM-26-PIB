import argparse
from contextlib import nullcontext
import copy
import json
import math
import shutil
from pathlib import Path

import torch
from torch import nn
import yaml

from .data import build_loaders
from .model import create_model
from .losses import PIBObjective
from .utils import (atomic_save, load_config, resolve_device, restore_rng,
                    rng_state, seed_everything, write_json)


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16.")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA.")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def learning_rate_multiplier(epoch, epochs, warmup):
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1, progress)))


def run_epoch(model, loader, device, optimizer=None, scaler=None, precision="fp32",
              objective=None, regularization_scale=1.0, gradient_clip=None):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "task", "pib", "path", "compression", "sufficiency")}
    correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                output = model(images, return_states=training)
            if training:
                values = objective(output, labels, regularization_scale)
                logits = output["logits"]
            else:
                logits = output
                values = {"loss": nn.functional.cross_entropy(logits.float(), labels)}
                values["task"] = values["loss"]
            loss = values["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check images, coefficients, and precision.")
            if training:
                scaler.scale(loss).backward()
                if gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    parameters = [p for group in optimizer.param_groups for p in group["params"]]
                    nn.utils.clip_grad_norm_(parameters, gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            batch_size = labels.numel()
            for key in totals:
                if key in values:
                    totals[key] += float(values[key].detach()) * batch_size
            correct += int((logits.argmax(-1) == labels).sum())
            count += batch_size
    if count == 0:
        raise ValueError("Cannot train or evaluate an empty split.")
    return {**{key: value / count for key, value in totals.items()},
            "top1": 100 * correct / count, "samples": count}


def run_training(config, output_dir, resume=None, device=None):
    config = copy.deepcopy(config)
    settings = config.setdefault("training", {})
    seed_everything(int(settings.get("seed", 0)))
    device = resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(config)
    if not {"train", "val"}.issubset(loaders):
        raise ValueError("Training requires separate train and validation splits.")
    num_classes = loaders["train"].dataset.num_classes
    declared_classes = config.setdefault("model", {}).get("num_classes", num_classes)
    if declared_classes != num_classes:
        raise ValueError(f"model.num_classes={declared_classes}, dataset has {num_classes} classes.")
    config["model"]["num_classes"] = num_classes
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint:
        for section in ("model", "loss", "data"):
            if checkpoint["config"].get(section, {}) != config.get(section, {}):
                raise ValueError(f"Resume {section} settings differ from the checkpoint.")
    model = create_model(config, pretrained=False if checkpoint else None).to(device)
    objective = PIBObjective(len(model.prompt_layers), **config.get("loss", {})).to(device)
    parameters = [parameter for parameter in list(model.parameters()) + list(objective.parameters())
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(settings.get("learning_rate", 1e-3)),
                                 weight_decay=float(settings.get("weight_decay", 1e-4)))
    epochs = int(settings.get("epochs", 100))
    warmup = int(settings.get("warmup_epochs", 10))
    if epochs < 1 or not 0 <= warmup <= epochs:
        raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs].")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: learning_rate_multiplier(epoch, epochs, warmup))
    precision = settings.get("precision", "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    start_epoch, best_top1 = 0, -math.inf
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        old_settings = checkpoint["config"].get("training", {})
        if (epochs, warmup) != (old_settings.get("epochs", 100), old_settings.get("warmup_epochs", 10)):
            factor = learning_rate_multiplier(checkpoint["epoch"] + 1, epochs, warmup)
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * factor
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_top1 = checkpoint["epoch"] + 1, checkpoint["best_top1"]
        restore_rng(checkpoint["rng"])
        for split, state in checkpoint.get("loader_rng", {}).items():
            if split in loaders and loaders[split].generator is not None:
                loaders[split].generator.set_state(state)
    if checkpoint and not (output_dir / "best.pt").exists():
        source_best = Path(resume).parent / "best.pt"
        if source_best.is_file():
            shutil.copyfile(source_best, output_dir / "best.pt")
        elif checkpoint["best_top1"] == run_epoch(model, loaders["val"], device)["top1"]:
            atomic_save(checkpoint, output_dir / "best.pt")
        else:
            raise ValueError("Resume needs the original best.pt alongside last.pt.")
    if start_epoch >= epochs:
        raise ValueError("The checkpoint has already completed the requested epochs.")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    trainable = sum(p.numel() for p in parameters)
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in objective.parameters())
    print(json.dumps({"device": str(device), "trainable_parameters": trainable,
                      "total_parameters": total, "start_epoch": start_epoch}), flush=True)
    history_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        regularization_scale = min(1.0, (epoch + 1) / max(1, int(settings.get("regularization_warmup", 10))))
        train_metrics = run_epoch(model, loaders["train"], device, optimizer, scaler, precision,
                                  objective, regularization_scale, settings.get("gradient_clip"))
        validation = run_epoch(model, loaders["val"], device, precision=precision)
        improved = validation["top1"] > best_top1
        best_top1 = max(best_top1, validation["top1"])
        scheduler.step()
        record = {"epoch": epoch + 1, "learning_rate": learning_rate,
                  "train": train_metrics, "val": validation, "best_val_top1": best_top1,
                  "regularization_scale": regularization_scale,
                  "routing_gates": objective.routing_logits.sigmoid().detach().cpu().tolist()
                  if objective.routing_logits is not None else None}
        print(json.dumps(record), flush=True)
        with history_path.open("a" if epoch else "w") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "format_version": 1, "epoch": epoch, "config": config,
            "model": model.state_dict(), "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_top1": best_top1, "rng": rng_state(),
            "loader_rng": {split: loader.generator.get_state()
                           for split, loader in loaders.items() if loader.generator is not None},
        }
        atomic_save(state, output_dir / "last.pt")
        if improved:
            atomic_save(state, output_dir / "best.pt")
    result = {"best_val_top1": best_top1, "epochs": epochs,
              "trainable_parameters": trainable, "total_parameters": total,
              "best_checkpoint": str(output_dir / "best.pt"),
              "last_checkpoint": str(output_dir / "last.pt")}
    write_json(result, output_dir / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train Prompted Information Bottlenecks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_training(load_config(args.config, args.set), args.output, args.resume, args.device)


if __name__ == "__main__":
    main()
