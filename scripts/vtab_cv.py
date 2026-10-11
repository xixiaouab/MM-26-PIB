"""Queue drivers: choose each task's 1,000-image-protocol configuration by five-fold cross-validation.

start   per task (benchmark vtab_png), the top-K configurations by mean last-epoch validation accuracy over
        at least three seeds among all PIB runs are trained on each of five folds (seed 0); then the finish
        driver is queued.
finish  per task, the configuration with the best mean last-epoch validation accuracy over the five folds
        is trained on all 1,000 images with five seeds and tested at the last epoch (identical earlier runs
        are reused); a report driver writes the round's report.
Round 1: top-8, folds over all 1,000 images (prefixes vtab_cv1 / vtab_p1000cv, report vtab_report_cv.md).
Round 2: top-20; smallNORB, whose val200 images show other object instances than train800, uses
val200-only folds (cvv, see vtab_cv_folds.py); the other tasks reuse round-1 fold runs (prefixes vtab_cv2 /
vtab_p1000cv2, report vtab_report_cv2.md). Round 3 repeats round 2 over the candidates including the SSF
runs of vtab_ad.py round f (prefixes vtab_cv3 / vtab_p1000cv3, report vtab_report_cv3.md).
The protocol-800 numbers are unaffected: that protocol selects on val200 by definition.
"""

import argparse
import json
from pathlib import Path

from benchmarks import tasks as benchmark_tasks
from queue_append import append_tasks
from vtab_ad import P, RUNS, candidates, cost, read_tasks, run
from vtab_cv_folds import VAL_ONLY

ROUNDS = {
    "1": {"top": "8", "prefix": "vtab_cv1", "final": "vtab_p1000cv", "report": "vtab_report_cv.md", "val_only": (),
          "folds": ["vtab_cv1"], "candidates": "e"},
    "2": {"top": "20", "prefix": "vtab_cv2", "final": "vtab_p1000cv2", "report": "vtab_report_cv2.md",
          "val_only": VAL_ONLY, "folds": ["vtab_cv1", "vtab_cv2"], "candidates": "e"},
    # Round 3: as round 2, over the candidates of vtab_ad.py round f (which adds the SSF runs).
    "3": {"top": "20", "prefix": "vtab_cv3", "final": "vtab_p1000cv3", "report": "vtab_report_cv3.md",
          "val_only": VAL_ONLY, "folds": ["vtab_cv1", "vtab_cv2", "vtab_cv3"], "candidates": "f"},
    # Round 4: as round 3, over the candidates of vtab_ad.py round g (which adds the augmentation variants).
    "4": {"top": "20", "prefix": "vtab_cv4", "final": "vtab_p1000cv4", "report": "vtab_report_cv4.md",
          "val_only": VAL_ONLY, "folds": ["vtab_cv1", "vtab_cv2", "vtab_cv3", "vtab_cv4"], "candidates": "g"},
    # Round 5: over the candidates of vtab_ad.py round h (flips for structured tasks, stronger crops).
    "5": {"top": "20", "prefix": "vtab_cv5", "final": "vtab_p1000cv5", "report": "vtab_report_cv5.md",
          "val_only": VAL_ONLY, "folds": ["vtab_cv1", "vtab_cv2", "vtab_cv3", "vtab_cv4", "vtab_cv5"], "candidates": "h"},
    # Round 6: over the candidates of vtab_ad.py round i (dihedral augmentation).
    "6": {"top": "20", "prefix": "vtab_cv6", "final": "vtab_p1000cv6", "report": "vtab_report_cv6.md",
          "val_only": VAL_ONLY, "folds": ["vtab_cv1", "vtab_cv2", "vtab_cv3", "vtab_cv4", "vtab_cv5", "vtab_cv6"],
          "candidates": "i"},
    # Round 7: over the candidates of vtab_ad.py round j (300-epoch variants).
    "7": {"top": "20", "prefix": "vtab_cv7", "final": "vtab_p1000cv7", "report": "vtab_report_cv7.md",
          "val_only": VAL_ONLY, "folds": [f"vtab_cv{k}" for k in range(1, 8)], "candidates": "j"},
    # Round 8: the top-50 candidates of the eight structured tasks (new fold runs only for those tasks); the
    # final selection still covers every task with all fold runs.
    "8": {"top": "50", "prefix": "vtab_cv8", "final": "vtab_p1000cv8", "report": "vtab_report_cv8.md",
          "val_only": VAL_ONLY, "folds": [f"vtab_cv{k}" for k in range(1, 9)], "candidates": "j",
          "tasks": ["clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori",
                    "smallnorb_azi_png", "smallnorb_ele_png"]},
}
P1000 = ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e", "vtab_p1000cv",
         "vtab_p1000cv2", "vtab_p1000ssf", "vtab_p1000cv3", "vtab_p1000g", "vtab_p1000cv4", "vtab_p1000h",
         "vtab_p1000cv5", "vtab_p1000i", "vtab_p1000cv6", "vtab_p1000j", "vtab_p1000cv7", "vtab_p1000cv8"]
PROTOCOLS = {"cv": [f"cv{fold}" for fold in range(5)], "cvv": [f"cvv{fold}" for fold in range(5)]}


def driver_name(round_, stage):
    return f"driver_vtab_cv_{stage}" if round_ == "1" else f"driver_vtab_cv{round_}_{stage}"


def output_path(round_, name):
    return f"{P}/{name}" if round_ == "1" else f"{P}/cv{round_}_{name}"


def start(round_, queue):
    spec = ROUNDS[round_]
    top = output_path(round_, f"cv_top{spec['top']}.json")
    run("collect.py", "--benchmark", "vtab_png", "--runs", RUNS, "--prefix", *candidates(spec["candidates"]), "--protocol", "800",
        "--methods", "pib", "--min-seeds", "3", "--metric", "last_val_top1", "--select", top, "--top", spec["top"])
    names = [task for task in benchmark_tasks("vtab_png") if task in spec.get("tasks", benchmark_tasks("vtab_png"))]
    groups = {"cv": [task for task in names if task not in spec["val_only"]],
              "cvv": [task for task in names if task in spec["val_only"]]}
    tasks = []
    for kind, members in groups.items():
        for protocol in PROTOCOLS[kind] if members else []:
            output = output_path(round_, f"q_cv_{protocol}.jsonl") if round_ != "1" else P + f"/q_cv_fold{protocol[2:]}.jsonl"
            run("make_queue.py", "--benchmark", "vtab_png_cv", "--stage", "refine", "--prefix", spec["prefix"],
                "--protocol", protocol, "--methods", "pib", "--tasks", *members, "--selection", top, "--seeds", "0",
                "--output", output)
            tasks += read_tasks(output)
    tasks = [task for task in tasks if not any(Path(RUNS, prefix, task["name"].split("/", 1)[1], "result.json").exists()
                                               for prefix in spec["folds"])]
    tasks.sort(key=lambda task: -cost(task))
    finish_driver = {"name": driver_name(round_, "finish"), "driver": True,
                     "cmd": f"cd scripts && python vtab_cv.py finish --round {round_} --queue {queue}"}
    append_tasks(queue, tasks + [finish_driver])
    print(f"appended {len(tasks)} cross-validation runs and the finish driver")


def finish(round_, queue):
    spec, selections = ROUNDS[round_], {}
    for kind in ("cv", "cvv"):
        output = output_path(round_, f"sel_{kind}.json")
        print(run("collect.py", "--benchmark", "vtab_png_cv", "--runs", RUNS, "--prefix", *spec["folds"],
                  "--protocols", *PROTOCOLS[kind], "--methods", "pib", "--min-seeds", "5", "--metric", "last_val_top1",
                  "--select", output))
        selections[kind] = json.loads(Path(output).read_text()).get("pib", {})
    chosen = {task: config for task, config in selections["cv"].items() if task not in spec["val_only"]}
    chosen.update({task: selections["cvv"][task] for task in spec["val_only"]})
    selection = output_path(round_, "sel_cv_p1000.json")
    Path(selection).write_text(json.dumps({"pib": chosen}, indent=2) + "\n")
    run("make_queue.py", "--benchmark", "vtab_png", "--stage", "final", "--prefix", spec["final"], "--protocol", "1000",
        "--methods", "pib", "--selection", selection, "--seeds", "0", "1", "2", "3", "4",
        "--output", output_path(round_, "q_cv_p1000.jsonl"))
    finals = [task for task in read_tasks(output_path(round_, "q_cv_p1000.jsonl"))
              if not any(Path(RUNS, prefix, task["name"].split("/", 1)[1], "result.json").exists() for prefix in P1000)]
    finals.sort(key=lambda task: -cost(task))
    report = (f"cd scripts && (echo '# VTAB-1k protocol 1000: configurations chosen by five-fold cross-validation"
              f" (round {round_})'; echo; echo '## Protocol 1000: 1,000 images, last checkpoint, 5 seeds';"
              f" python collect.py --benchmark vtab_png --runs {RUNS} --prefix {' '.join(P1000)}"
              f" --protocol 1000 --methods pib --selection {selection}) > {P}/{spec['report']}")
    append_tasks(queue, finals + [{"name": driver_name(round_, "report"), "driver": True, "cmd": report}])
    print(f"appended {len(finals)} protocol-1000 runs and the report driver")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("start", "finish"))
    parser.add_argument("--round", default="1", choices=sorted(ROUNDS))
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    {"start": start, "finish": finish}[args.stage](args.round, args.queue)


if __name__ == "__main__":
    main()
