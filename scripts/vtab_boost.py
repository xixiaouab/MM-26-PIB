"""Queue driver, stage 2 of the VTAB-1k push (validation only).

Top-3 configurations per task over the AdamW and SGD grids get seeds 1-2; each task's top
configuration also gets variants with seeds 0-2: CLS+GAP features, 300 epochs, PIB loss scales
0.1/0.3, and horizontal flips on the natural-image tasks. The tasks are appended to --queue,
followed by the stage-3 driver (vtab_final_stage.py).
"""

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks import expected_cost
from queue_append import append_tasks

P = "/lustre/orion/csc662/proj-shared/xxiao/PIB"
POOL = ["vtab_s1", "vtab_s1x", "vtab_sgd1"]
EXISTING = POOL + ["vtab_f1", "vtab_r1", "vtab_v1"]
NATURAL_FLIP = ["cifar", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "sun397"]


def run(*args):
    result = subprocess.run([P + "/envs/pib/bin/python", *args], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-2000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    run("collect.py", "--benchmark", "vtab", "--prefix", *POOL, "--select", P + "/boost_top3.json", "--top", "3")
    run("collect.py", "--benchmark", "vtab", "--prefix", *POOL, "--select", P + "/boost_top1.json")
    outputs = []

    def queue(name, *extra):
        out = P + f"/q_boost_{name}.jsonl"
        run("make_queue.py", "--benchmark", "vtab", "--stage", "refine", "--prefix", "vtab_b1", "--output", out, *extra)
        outputs.append(out)

    queue("seeds", "--selection", P + "/boost_top3.json", "--seeds", "1", "2")
    top1 = ("--selection", P + "/boost_top1.json", "--seeds", "0", "1", "2")
    queue("gap", *top1, "--set", 'model.features="cls_gap"', "--tag", "gap")
    queue("ep300", *top1, "--set", "training.epochs=300", "--tag", "ep300")
    queue("flip", *top1, "--tasks", *NATURAL_FLIP, "--set", 'data.augment="flip"', "--tag", "flip")
    for scale in ("0.1", "0.3"):
        queue(f"ls{scale}", *top1, "--methods", "pib", "--loss-scale", scale)
    runs = Path(P + "/runs")

    def exists(name):
        rest = name.split("/", 1)[1]
        return any((runs / prefix / rest / "result.json").exists() for prefix in EXISTING)

    tasks = [json.loads(line) for out in outputs for line in Path(out).read_text().splitlines()]
    tasks = [t for t in tasks if not exists(t["name"])]
    cost = lambda t: (expected_cost(t["name"].split("/")[2], t["set"]["model.prompt_length"])
                      * t["set"].get("training.epochs", 100))
    tasks.sort(key=lambda t: -cost(t))
    driver = {"name": "driver_vtab_final", "driver": True,
              "cmd": f"cd scripts && python vtab_final_stage.py --queue {args.queue}"}
    append_tasks(args.queue, tasks + [driver])
    print(f"appended {len(tasks)} boost runs and the final-stage driver")


if __name__ == "__main__":
    main()
