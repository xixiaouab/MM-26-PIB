"""Compare variants of each task's selected configuration on validation accuracy only.

For every method and task, the base configuration from a selection JSON and each variant tag found
under --prefix (same prompt length, learning rate and weight decay; tag = base tag + "-" + variant)
are scored by the mean validation accuracy over seeds, and summarized with the benchmark's mean of
group means. Test accuracy is never read.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks import BENCHMARKS
from collect import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--base-prefix", nargs="+", required=True, help="prefixes holding the base runs")
    parser.add_argument("--prefix", nargs="+", required=True, help="prefixes holding the variant runs")
    parser.add_argument("--runs", default="/lustre/orion/csc662/proj-shared/xxiao/PIB/runs")
    parser.add_argument("--protocol", default="800")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    groups = BENCHMARKS[args.benchmark]["groups"]
    scores = defaultdict(lambda: defaultdict(list))  # (method, variant) -> task -> vals
    rows = load(Path(args.runs), args.base_prefix + args.prefix, args.protocol)
    for row in rows:
        chosen = selection.get(row["method"], {}).get(row["task"])
        if not chosen or (int(row["length"]), float(row["lr"]), float(row["wd"])) != (chosen["length"], chosen["lr"], chosen["wd"]):
            continue
        base_tag, tag = chosen.get("tag") or "", row.get("tag") or ""
        if tag == base_tag:
            variant = "base"
        elif tag.startswith(base_tag + "-" if base_tag else ""):
            variant = tag[len(base_tag) + 1:] if base_tag else tag
        else:
            continue
        scores[(row["method"], variant)][row["task"]].append(row["best_val_top1"])
    variants = sorted({v for _, v in scores}, key=lambda v: (v != "base", v))
    for method in sorted({m for m, _ in scores}):
        print(f"\n## {method} (validation accuracy, mean over seeds)\n")
        print("| task | " + " | ".join(variants) + " |\n|---|" + "---|" * len(variants))
        summary = {}
        for variant in variants:
            task_scores = scores.get((method, variant), {})
            means = {task: statistics.mean(v) for task, v in task_scores.items()}
            complete = all(task in means for names in groups.values() for task in names)
            summary[variant] = (statistics.mean(statistics.mean(means[t] for t in names) for names in groups.values())
                                if complete else None)
            scores[(method, variant)] = means
        for names in groups.values():
            for task in names:
                cells = [f"{scores[(method, v)][task]:.1f}" if task in scores[(method, v)] else "-" for v in variants]
                print(f"| {task} | " + " | ".join(cells) + " |")
        print("| **mean of group means** | " + " | ".join(f"**{summary[v]:.2f}**" if summary[v] is not None else "-"
                                                      for v in variants) + " |")


if __name__ == "__main__":
    main()
