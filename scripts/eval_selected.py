"""Write a queue that evaluates the test split of already trained runs of the selected configurations.

For each method and task in the selection JSON (collect.py --select), every seed of the chosen
configuration found under --prefix gets one queue task that evaluates its best-validation checkpoint
on the test split and adds test_top1 to the run's result.json. Selection never looks at test data.
"""

import argparse
import json
from pathlib import Path

from collect import is_chosen, load

RUNS = "/lustre/orion/csc662/proj-shared/xxiao/PIB/runs"


def chosen_runs(rows, selection):
    return (row for row in rows if is_chosen(row, selection))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--prefix", nargs="+", required=True)
    parser.add_argument("--protocol", default="800")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", default=RUNS)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    rows = list(chosen_runs(load(Path(args.runs), args.prefix, args.protocol), selection))
    tasks = [{"name": "test_" + Path(row["dir"]).relative_to(args.runs).as_posix(),
              "cmd": f"python scripts/eval_run.py --run {row['dir']}"} for row in rows]
    counts = {}
    for row in rows:
        counts[(row["method"], row["task"])] = counts.get((row["method"], row["task"]), 0) + 1
    for (method, task), count in sorted(counts.items()):
        print(f"{method:4s} {task:22s} {count} runs")
    Path(args.output).write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"{len(tasks)} evaluations -> {args.output}")


if __name__ == "__main__":
    main()
