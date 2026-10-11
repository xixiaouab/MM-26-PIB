"""Run one queued experiment: train, then optionally evaluate a checkpoint on the test split.

The task is a JSON object:
    {"name": "...", "config": "configs/vtab_vit_b16.yaml", "set": {"training.learning_rate": 0.001, ...},
     "eval": "best" | "last" | null}
"best" is the checkpoint with the highest validation accuracy, "last" the final one.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from pib.evaluate import evaluate_checkpoint
from pib.train import TrainingPreempted, run_training
from pib.utils import load_config, write_json

PREEMPTED = 75  # exit code: stopped at the deadline, resume later from last.pt


def run(task, output, deadline=None, eval_margin=300):
    """Train (resuming from last.pt if a previous attempt was preempted), then evaluate."""
    output = Path(output)
    start = time.time()
    overrides = [f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}"
                 for key, value in task.get("set", {}).items()]
    config = load_config(task["config"], overrides)
    last = output / "last.pt"
    resume = last if last.exists() else None
    epochs = int(config.get("training", {}).get("epochs", 100))
    if resume is not None and torch.load(last, map_location="cpu", weights_only=True)["epoch"] + 1 >= epochs:
        summary = json.loads((output / "summary.json").read_text())  # training finished; only evaluation is left
    else:
        summary = run_training(config, output, resume=resume,
                               deadline=None if deadline is None else deadline - eval_margin)
    result = {"name": task["name"], "task": task, "best_val_top1": summary["best_val_top1"],
              "trainable_parameters": summary["trainable_parameters"], "train_seconds": time.time() - start}
    history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    result["last_val_top1"] = history[-1]["val"]["top1"]
    result["best_epoch"] = max(history, key=lambda row: (row["val"]["top1"], -row["epoch"]))["epoch"]
    if task.get("eval"):
        metrics = evaluate_checkpoint(output / f"{task['eval']}.pt", split="test", output=output / "test")
        result.update(test_checkpoint=task["eval"], test_top1=metrics["top1"], test_samples=metrics["samples"])
    result["seconds"] = time.time() - start
    write_json(result, output / "result.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="task as a JSON string")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deadline = float(os.environ["PIB_DEADLINE"]) if os.environ.get("PIB_DEADLINE") else None
    try:
        run(json.loads(args.task), args.output, deadline)
    except TrainingPreempted as stop:
        print(f"preempted: {stop}", flush=True)
        sys.exit(PREEMPTED)


if __name__ == "__main__":
    main()
