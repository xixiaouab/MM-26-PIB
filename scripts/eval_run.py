"""Evaluate one finished run's best-validation checkpoint on the test split and record it in result.json."""

import argparse
import json
from pathlib import Path

from pib.evaluate import evaluate_checkpoint
from pib.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="best", choices=("best", "last"))
    args = parser.parse_args()
    run = Path(args.run)
    result = json.loads((run / "result.json").read_text())
    metrics = evaluate_checkpoint(run / f"{args.checkpoint}.pt", split="test", output=run / "test")
    result.update(test_checkpoint=args.checkpoint, test_top1=metrics["top1"], test_samples=metrics["samples"])
    write_json(result, run / "result.json")


if __name__ == "__main__":
    main()
