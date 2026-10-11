"""Summarize runs of a benchmark.

--select OUT  For each method and task, choose the configuration (prompt length, learning rate, weight
              decay, tag) with the highest mean validation accuracy over seeds; ties go to the shorter
              prompt, then the smaller learning rate and weight decay. Writes the selection JSON for
              make_queue.py --stage final. Test accuracy is never read here.
--metric      best_val_top1 (default; protocols that test the best-validation checkpoint) or
              last_val_top1 (protocols that test the last checkpoint).
(default)     Test accuracy of the final runs per task (mean and std over seeds), group means and the
              benchmark mean, next to the paper's PIB numbers. For VTAB-1k the benchmark mean is the
              mean of the three group means (as in the paper); the plain task mean is shown too.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks import BENCHMARKS

NAME = re.compile(r"(?P<method>[^/]+)/(?P<task>[^/]+)/L(?P<length>\d+)_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)"
                  r"_p(?P<protocol>[^_]+)_s(?P<seed>\d+)(?:_(?P<tag>.+))?$")
STANDARD_KEYS = {"data.manifest", "model.prompt_length", "training.learning_rate", "training.weight_decay",
                 "training.seed"}


def load(runs, prefixes, protocol=None):
    rows = []
    for prefix in prefixes:
        for path in sorted((runs / prefix).glob("*/*/*/result.json")):
            match = NAME.match(str(path.parent.relative_to(runs / prefix)))
            if match and (protocol is None or match["protocol"] == protocol):
                result = json.loads(path.read_text())
                result["spec"] = result.pop("task", {})  # the run's task spec; "task" is the dataset name
                rows.append({**result, **match.groupdict(), "dir": str(path.parent)})
    return rows


def rank_configs(rows, min_seeds=1, metric="best_val_top1"):
    """Per (method, task), configurations sorted by mean validation accuracy over seeds (best first)."""
    configs = defaultdict(list)
    for row in rows:
        key = (row["method"], row["task"], int(row["length"]), float(row["lr"]), float(row["wd"]), row.get("tag") or "")
        configs[key].append(row)
    ranked = defaultdict(list)
    for (method, task, length, lr, wd, tag), members in configs.items():
        if len(members) < min_seeds:
            continue
        val = statistics.mean(row[metric] for row in members)
        settings = members[0].get("spec", {}).get("set", {})
        extra = {k: v for k, v in settings.items() if k not in STANDARD_KEYS
                 and not (method == "vpt" and k.startswith("loss."))}
        ranked[(method, task)].append(((-val, length, lr, wd, tag), {
            "length": length, "lr": lr, "wd": wd, "tag": tag, "set": extra, "val": val, "seeds": len(members)}))
    return {key: [config for _, config in sorted(items, key=lambda item: item[0])] for key, items in ranked.items()}


def select(rows, top=1, min_seeds=1, metric="best_val_top1"):
    """The best configuration per method and task (a list of the best `top` when top > 1)."""
    selection = defaultdict(dict)
    for (method, task), configs in sorted(rank_configs(rows, min_seeds, metric).items()):
        selection[method][task] = configs[0] if top == 1 else configs[:top]
    return selection


def is_chosen(row, selection):
    chosen = selection.get(row["method"], {}).get(row["task"])
    return bool(chosen) and (int(row["length"]), float(row["lr"]), float(row["wd"]), row.get("tag") or "") == (
        chosen["length"], chosen["lr"], chosen["wd"], chosen.get("tag") or "")


def report(rows, benchmark, selection=None):
    """Test accuracy per task; with a selection, only the runs of each task's chosen configuration count."""
    spec = BENCHMARKS[benchmark]
    tests = defaultdict(list)
    for row in rows:
        if "test_top1" in row and (selection is None or is_chosen(row, selection)):
            tests[(row["method"], row["task"])].append(row["test_top1"])
    for method in sorted({m for m, _ in tests}):
        print(f"\n## {benchmark} / {method}\n\n| task | test (mean ± std) | seeds | paper PIB |\n|---|---|---|---|")
        group_means, task_means = {}, []
        for group, names in spec["groups"].items():
            means = []
            for task in names:
                values = tests.get((method, task), [])
                if not values:
                    print(f"| {task} | missing | 0 | {spec['paper'][task]} |")
                    continue
                mean, std = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
                means.append(mean)
                task_means.append(mean)
                print(f"| {task} | {mean:.2f} ± {std:.2f} | {len(values)} | {spec['paper'][task]} |")
            if len(means) == len(names):
                group_means[group] = statistics.mean(means)
                paper = statistics.mean(spec["paper"][t] for t in names)
                print(f"| **{group} mean** | **{group_means[group]:.2f}** | | {paper:.2f} |")
        if len(group_means) == len(spec["groups"]) and len(group_means) > 1:
            paper = statistics.mean(statistics.mean(spec["paper"][t] for t in names) for names in spec["groups"].values())
            print(f"| **mean of group means** | **{statistics.mean(group_means.values()):.2f}** | | {paper:.2f} |")
            print(f"| task mean | {statistics.mean(task_means):.2f} | | {statistics.mean(spec['paper'].values()):.2f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--runs", default="/lustre/orion/csc662/proj-shared/xxiao/PIB/runs")
    parser.add_argument("--prefix", nargs="+", required=True, help="one or more run prefixes")
    parser.add_argument("--protocol", help="only runs of this protocol (e.g. 800 or 1000)")
    parser.add_argument("--protocols", nargs="+", help="only runs of these protocols (e.g. the five folds cv0 ... cv4)")
    parser.add_argument("--select", metavar="OUTPUT_JSON")
    parser.add_argument("--top", type=int, default=1, help="with --select: keep the best K configurations per task")
    parser.add_argument("--min-seeds", type=int, default=1, help="with --select: ignore configurations with fewer seeds")
    parser.add_argument("--selection", help="report only the runs of the configurations in this selection JSON")
    parser.add_argument("--metric", choices=("best_val_top1", "last_val_top1"), default="best_val_top1",
                        help="with --select: validation accuracy of the best (default) or the last epoch")
    parser.add_argument("--methods", nargs="+", help="only runs of these methods")
    args = parser.parse_args()
    rows = [row for row in load(Path(args.runs), args.prefix, args.protocol)
            if (not args.methods or row["method"] in args.methods)
            and (not args.protocols or row["protocol"] in args.protocols)]
    print(f"{len(rows)} finished runs under {', '.join(args.prefix)}")
    if args.select:
        selection = select(rows, args.top, args.min_seeds, args.metric)
        Path(args.select).write_text(json.dumps(selection, indent=2) + "\n")
        for method, chosen_tasks in selection.items():
            for task, chosen in chosen_tasks.items():
                for config in chosen if isinstance(chosen, list) else [chosen]:
                    print(f"{method:4s} {task:22s} L={config['length']:<4d} lr={config['lr']:<8g} wd={config['wd']:<8g} "
                          f"{config['tag'] or '-':6s} val={config['val']:.1f} ({config['seeds']} seeds)")
    else:
        report(rows, args.benchmark, json.loads(Path(args.selection).read_text()) if args.selection else None)


if __name__ == "__main__":
    main()
