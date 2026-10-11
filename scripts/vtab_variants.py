"""Queue driver starting a variants round: variants of every task's current configuration.

On benchmark vtab_png (19 tasks; CIFAR-100, SVHN and smallNORB from their original pixels), each task's
configuration is selected among all earlier PIB runs with at least three seeds, by best-epoch and by
last-epoch validation accuracy, and those configurations are queued with each variant of the round (seeds
0-2), followed by the round's final stage (vtab_ad.py final --round R). Round e (default): label smoothing
0.1 on every task (prefix vtab_e1, tag smooth0.1). Round g: horizontal flips and mild random resized crops
on the natural and specialized tasks whose labels are invariant to them (prefix vtab_g1, tags flip / rrc).
Round h: flips on the structured tasks whose labels survive mirroring (counts, distances, elevation) and
stronger crops on the natural and specialized tasks (prefix vtab_h1). Round i: the eight rotations and
reflections of the square for tissue, satellite, retina and texture images (prefix vtab_i1, tag d4).
Round j: 300 epochs for the current configurations of the tasks with augmentations (prefix vtab_j1).
"""

import argparse
import json
from pathlib import Path

from queue_append import append_tasks
from vtab_ad import ROUNDS, RUNS, candidates, cost, driver, files, read_tasks, run

LABEL_SMOOTHING = {"smooth0.1": {"loss.label_smoothing": 0.1}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--round", default="e", choices=("e", "g", "h", "i", "j"))
    args = parser.parse_args()
    round_, outputs = args.round, []
    spec = ROUNDS[round_]
    earlier = [prefix for prefix in candidates(round_) if prefix != spec["search"]]
    # Variant groups: (tasks or None for all, {tag: overrides}); rounds e and g use a single group.
    groups = spec.get("variant_groups") or [{"tasks": spec.get("variant_tasks"),
                                             "sets": spec.get("variant_sets", LABEL_SMOOTHING)}]
    for metric in ("best_val_top1", "last_val_top1"):
        selection = files(round_, f"sel_current_{metric}.json")
        run("collect.py", "--benchmark", spec["benchmark"], "--runs", RUNS, "--prefix", *earlier, "--protocol", "800",
            "--methods", "pib", "--min-seeds", "3", "--metric", metric, "--select", selection)
        for number, group in enumerate(groups):
            for tag, overrides in group["sets"].items():
                name = f"q_smooth_{metric}.jsonl" if round_ == "e" else f"q_{number}_{tag}_{metric}.jsonl"
                output = files(round_, name)
                run("make_queue.py", "--benchmark", spec["benchmark"], "--stage", "refine", "--prefix", spec["search"],
                    "--methods", "pib", *(["--tasks", *group["tasks"]] if group["tasks"] else []),
                    "--selection", selection, "--seeds", "0", "1", "2",
                    "--set", *[f"{key}={json.dumps(value)}" for key, value in overrides.items()], "--tag", tag,
                    "--output", output)
                outputs.append(output)
    seen, tasks = set(), []
    for output in outputs:
        for task in read_tasks(output):
            rest = task["name"].split("/", 1)[1]
            if rest not in seen and not Path(RUNS, spec["search"], rest, "result.json").exists():
                seen.add(rest)
                tasks.append(task)
    tasks.sort(key=lambda task: -cost(task))
    append_tasks(args.queue, tasks + [driver("final", round_, args.queue)])
    print(f"appended {len(tasks)} variant runs and the final driver")


if __name__ == "__main__":
    main()
