"""Queue drivers for the VTAB-1k push with LoRA / adapter modules (PIB only, validation-only selection).

Each stage runs as a "driver" task of the queue (scripts/queue_worker.py) and appends the next stage:
  search  seed 0 for every module setting x prompt length x learning rate x weight decay (prefix vtab_ad1)
  refine  seeds 1-2 for the top-3 configurations by best-epoch and by last-epoch validation accuracy, and
          variants of the top configuration (PIB loss scale 0.3 / 0.1, CLS+GAP features) with seeds 0-2
          (prefix vtab_ad2)
  final   protocol 800: per task, the configuration with the best mean best-epoch validation accuracy over
          at least three seeds among all PIB runs; its train800 runs are tested at the best-validation
          checkpoint. Protocol 1000: per task, the configuration with the best mean last-epoch validation
          accuracy, trained on all 1,000 images with five seeds and tested at the last epoch (prefix
          vtab_p1000ad). A report driver then writes $PIB_ROOT/vtab_report_ad.md.
Round 2 (--round 2) searches larger modules and 300-epoch training on the structured and specialized
tasks and SVHN (prefixes vtab_ad3 / vtab_ad4 / vtab_p1000ad2, report vtab_report_ad2.md); its final
selection considers every earlier PIB run, and protocol-1000 runs that already exist are reused.
Round c (--round c) repeats search, refinement and selection for CIFAR-100 on its original pixels
(benchmark vtab_png, task cifar_png; prefixes vtab_c1 / vtab_c2 / vtab_p1000c, report vtab_report_c.md),
choosing only among runs on that data; round s does the same for SVHN (task svhn_png; prefixes vtab_sv1 /
vtab_sv2 / vtab_p1000sv, report vtab_report_sv.md) and round n for the two smallNORB tasks (vtab_n1 /
vtab_n2 / vtab_p1000n, report vtab_report_n.md).
Start round R with a queue holding only driver("search", R, <queue>).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmarks import expected_cost
from queue_append import append_tasks

P = os.environ.get("PIB_ROOT", "/lustre/orion/csc662/proj-shared/xxiao/PIB")
RUNS = P + "/runs"
ALL = ["qkv", "proj", "fc1", "fc2"]
EARLIER = ["vtab_s1", "vtab_s1x", "vtab_f1", "vtab_r1", "vtab_v1", "vtab_sgd1", "vtab_b1"]
ROUNDS = {
    "1": {"search": "vtab_ad1", "refine": "vtab_ad2", "final": "vtab_p1000ad", "report": "vtab_report_ad.md",
          "tasks": None, "lengths": ["1", "10"], "lrs": ["0.0005", "0.001", "0.003", "0.01"], "wds": ["0.0001", "0.01"],
          "epochs": [100], "variants": True, "reuse": [],
          "modules": {"lora8": {"model.lora": {"rank": 8, "targets": ["qkv"]}},
                      "lora8all": {"model.lora": {"rank": 8, "targets": ALL}},
                      "ad32": {"model.adapter": {"dim": 32, "scale": 0.1}},
                      "lora8-ad32": {"model.lora": {"rank": 8, "targets": ["qkv"]},
                                     "model.adapter": {"dim": 32, "scale": 0.1}}}},
    "2": {"search": "vtab_ad3", "refine": "vtab_ad4", "final": "vtab_p1000ad2", "report": "vtab_report_ad2.md",
          "tasks": ["svhn", "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count",
                    "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"],
          "lengths": ["1"], "lrs": ["0.001", "0.003", "0.01"], "wds": ["0.0001", "0.01"],
          "epochs": [100, 300], "variants": False, "reuse": ["vtab_p1000ad"],
          "modules": {"lora16all": {"model.lora": {"rank": 16, "targets": ALL}},
                      "ad64s1": {"model.adapter": {"dim": 64, "scale": 1.0}},
                      "lora8all-ad32": {"model.lora": {"rank": 8, "targets": ALL},
                                        "model.adapter": {"dim": 32, "scale": 0.1}}}},
    # Module entries may carry their own grids: {"set": {...}, "lengths": [...], "lrs": [...], "wds": [...]}.
    "c": {"benchmark": "vtab_png", "search": "vtab_c1", "refine": "vtab_c2", "final": "vtab_p1000c",
          "report": "vtab_report_c.md", "tasks": ["cifar_png"], "lengths": ["1", "10"], "lrs": ["0.001", "0.003", "0.01"],
          "wds": ["0.0001", "0.01"], "epochs": [100], "variants": True, "reuse": [], "candidates": ["vtab_c1", "vtab_c2"],
          "modules": {"sgd": {"set": {"training.optimizer": "sgd", "model.prompt_init": "xavier"},
                              "lengths": ["1", "5", "10", "50"], "lrs": ["0.125", "0.25", "0.625", "1.25", "2.5"],
                              "wds": ["0.01", "0.001"]},
                      "": {"set": {}, "lengths": ["1", "5", "10", "50"], "lrs": ["0.001", "0.003", "0.01", "0.03"],
                           "wds": ["0.0001"]},
                      "ad32": {"model.adapter": {"dim": 32, "scale": 0.1}},
                      "lora8": {"model.lora": {"rank": 8, "targets": ["qkv"]}}}},
    "s": {"benchmark": "vtab_png", "search": "vtab_sv1", "refine": "vtab_sv2", "final": "vtab_p1000sv",
          "report": "vtab_report_sv.md", "tasks": ["svhn_png"], "lengths": ["1", "10"], "lrs": ["0.001", "0.003", "0.01"],
          "wds": ["0.0001", "0.01"], "epochs": [100], "variants": True, "reuse": [], "candidates": ["vtab_sv1", "vtab_sv2"],
          "modules": {"sgd": {"set": {"training.optimizer": "sgd", "model.prompt_init": "xavier"},
                              "lengths": ["1", "5", "10", "50"], "lrs": ["0.125", "0.25", "0.625", "1.25", "2.5"],
                              "wds": ["0.01", "0.001"]},
                      "": {"set": {}, "lengths": ["1", "5", "10", "50"], "lrs": ["0.001", "0.003", "0.01", "0.03"],
                           "wds": ["0.0001"]},
                      "ad32": {"model.adapter": {"dim": 32, "scale": 0.1}},
                      "lora8": {"model.lora": {"rank": 8, "targets": ["qkv"]}},
                      "lora8all": {"model.lora": {"rank": 8, "targets": ALL}}}},
    "n": {"benchmark": "vtab_png", "search": "vtab_n1", "refine": "vtab_n2", "final": "vtab_p1000n",
          "report": "vtab_report_n.md", "tasks": ["smallnorb_azi_png", "smallnorb_ele_png"], "lengths": ["1", "10"],
          "lrs": ["0.0005", "0.001", "0.003"], "wds": ["0.0001", "0.01"], "epochs": [100], "variants": True, "reuse": [],
          "candidates": ["vtab_n1", "vtab_n2"],
          "modules": {"sgd": {"set": {"training.optimizer": "sgd", "model.prompt_init": "xavier"},
                              "lengths": ["10", "50", "100", "200"], "lrs": ["0.0625", "0.125", "0.25", "0.625"],
                              "wds": ["0.01", "0.001"]},
                      "": {"set": {}, "lengths": ["10", "50", "100", "200"], "lrs": ["0.001", "0.003", "0.01"],
                           "wds": ["0.0001"]},
                      "lora8all": {"model.lora": {"rank": 8, "targets": ALL}},
                      "lora8-ad32": {"model.lora": {"rank": 8, "targets": ["qkv"]},
                                     "model.adapter": {"dim": 32, "scale": 0.1}},
                      "ad32": {"model.adapter": {"dim": 32, "scale": 0.1}}}},
    # Round e (scripts/vtab_variants.py starts it): label-smoothing variants of every task's current
    # configuration on vtab_png, then the final stage over every PIB run (lossless tasks by their *_png names).
    "e": {"benchmark": "vtab_png", "search": "vtab_e1", "refine": "vtab_e1", "final": "vtab_p1000e",
          "report": "vtab_report_e.md", "tasks": None, "variants": False, "modules": {},
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2",
                                   "vtab_sv1", "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1"]},
    # Round f: SSF (per-channel scale and shift) alone and with an adapter, on all 19 vtab_png tasks; its final
    # selection considers every PIB run (protocol 1000 is then re-chosen by cross-validation, vtab_cv.py round 3).
    "f": {"benchmark": "vtab_png", "search": "vtab_ssf1", "refine": "vtab_ssf2", "final": "vtab_p1000ssf",
          "report": "vtab_report_ssf.md", "tasks": None, "lengths": ["1", "10"], "lrs": ["0.001", "0.003", "0.01", "0.03"],
          "wds": ["0.0001", "0.01"], "epochs": [100], "variants": True,
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e",
                    "vtab_p1000cv", "vtab_p1000cv2"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2", "vtab_sv1",
                                   "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1", "vtab_ssf1", "vtab_ssf2"],
          "modules": {"ssf": {"model.ssf": True},
                      "ssf-ad32": {"model.ssf": True, "model.adapter": {"dim": 32, "scale": 0.1}}}},
    # Round g (scripts/vtab_variants.py --round g): augmentation variants of the current configurations of the
    # natural and specialized tasks whose labels survive flips and crops (not SVHN digits, not the geometric
    # structured tasks), then the final stage over every PIB run.
    "g": {"benchmark": "vtab_png", "search": "vtab_g1", "refine": "vtab_g1", "final": "vtab_p1000g",
          "report": "vtab_report_g.md", "tasks": None, "variants": False, "modules": {},
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e",
                    "vtab_p1000cv", "vtab_p1000cv2", "vtab_p1000ssf", "vtab_p1000cv3"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2", "vtab_sv1",
                                   "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1", "vtab_ssf1", "vtab_ssf2", "vtab_g1"],
          "variant_tasks": ["cifar_png", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "sun397",
                            "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"],
          "variant_sets": {"flip": {"data.augment": "flip"},
                           "rrc": {"data.augment": "rrc", "data.crop_scale": [0.5, 1.0]}}},
    # Round h: mirrored images for the structured tasks whose labels survive a horizontal flip (object counts,
    # distances, elevation; not positions, orientations or azimuths), and stronger random resized crops for the
    # natural and specialized tasks of round g.
    "h": {"benchmark": "vtab_png", "search": "vtab_h1", "refine": "vtab_h1", "final": "vtab_p1000h",
          "report": "vtab_report_h.md", "tasks": None, "variants": False, "modules": {},
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e",
                    "vtab_p1000cv", "vtab_p1000cv2", "vtab_p1000ssf", "vtab_p1000cv3", "vtab_p1000g", "vtab_p1000cv4"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2", "vtab_sv1",
                                   "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1", "vtab_ssf1", "vtab_ssf2", "vtab_g1",
                                   "vtab_h1"],
          "variant_groups": [
              {"tasks": ["clevr_count", "clevr_dist", "dmlab", "kitti", "smallnorb_ele_png"],
               "sets": {"flip": {"data.augment": "flip"}}},
              {"tasks": ["cifar_png", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "sun397",
                         "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"],
               "sets": {"rrc25": {"data.augment": "rrc", "data.crop_scale": [0.25, 1.0]}}}]},
    # Round i: the eight rotations / reflections of the square for images without an up direction.
    "i": {"benchmark": "vtab_png", "search": "vtab_i1", "refine": "vtab_i1", "final": "vtab_p1000i",
          "report": "vtab_report_i.md", "tasks": None, "variants": False, "modules": {},
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e",
                    "vtab_p1000cv", "vtab_p1000cv2", "vtab_p1000ssf", "vtab_p1000cv3", "vtab_p1000g", "vtab_p1000cv4",
                    "vtab_p1000h", "vtab_p1000cv5"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2", "vtab_sv1",
                                   "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1", "vtab_ssf1", "vtab_ssf2", "vtab_g1",
                                   "vtab_h1", "vtab_i1"],
          "variant_groups": [{"tasks": ["patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy", "dtd"],
                              "sets": {"d4": {"data.augment": "d4"}}}]},
    # Round j: 300 training epochs for the current configurations of the 15 tasks with label-preserving
    # augmentations (augmented data usually needs longer training).
    "j": {"benchmark": "vtab_png", "search": "vtab_j1", "refine": "vtab_j1", "final": "vtab_p1000j",
          "report": "vtab_report_j.md", "tasks": None, "variants": False, "modules": {},
          "reuse": ["vtab_p1000ad", "vtab_p1000ad2", "vtab_p1000c", "vtab_p1000sv", "vtab_p1000n", "vtab_p1000e",
                    "vtab_p1000cv", "vtab_p1000cv2", "vtab_p1000ssf", "vtab_p1000cv3", "vtab_p1000g", "vtab_p1000cv4",
                    "vtab_p1000h", "vtab_p1000cv5", "vtab_p1000i", "vtab_p1000cv6"],
          "candidates": EARLIER + ["vtab_ad1", "vtab_ad2", "vtab_ad3", "vtab_ad4", "vtab_c1", "vtab_c2", "vtab_sv1",
                                   "vtab_sv2", "vtab_n1", "vtab_n2", "vtab_e1", "vtab_ssf1", "vtab_ssf2", "vtab_g1",
                                   "vtab_h1", "vtab_i1", "vtab_j1"],
          "variant_groups": [{"tasks": ["cifar_png", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet",
                                        "sun397", "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy",
                                        "clevr_count", "clevr_dist", "dmlab", "kitti", "smallnorb_ele_png"],
                              "sets": {"ep300": {"training.epochs": 300}}}]},
}


def candidates(round_):
    """PIB run prefixes the final selection of a round considers: earlier campaigns and rounds up to this one."""
    if "candidates" in ROUNDS[round_]:
        return ROUNDS[round_]["candidates"]
    rounds = [r for r in ROUNDS if r.isdigit() and int(r) <= int(round_)]
    return EARLIER + [ROUNDS[r][stage] for r in rounds for stage in ("search", "refine")]


def run(*args):
    result = subprocess.run([P + "/envs/pib/bin/python", *args], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-2000:])
    return result.stdout


def read_tasks(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cost(task):
    return (expected_cost(task["name"].split("/")[2], task["set"]["model.prompt_length"])
            * task["set"].get("training.epochs", 100))


def driver(stage, round_, queue):
    name = f"driver_vtab_ad_{stage}" if round_ == "1" else f"driver_vtab_ad{round_}_{stage}"
    return {"name": name, "driver": True, "cmd": f"cd scripts && python vtab_ad.py {stage} --round {round_} --queue {queue}"}


def files(round_, name):
    return f"{P}/{'' if round_ == '1' else 'r' + round_ + '_'}{name}"


def search(round_, queue):
    spec, tasks = ROUNDS[round_], []
    for tag, module in spec["modules"].items():
        grid = module if "set" in module else {"set": module}
        for epochs in spec["epochs"]:
            output = files(round_, f"q_ad_search_{tag or 'plain'}_ep{epochs}.jsonl")
            overrides = {**grid["set"], **({"training.epochs": epochs} if epochs != 100 else {})}
            run("make_queue.py", "--benchmark", spec.get("benchmark", "vtab"), "--stage", "search",
                "--prefix", spec["search"], "--methods", "pib", *(["--tasks", *spec["tasks"]] if spec["tasks"] else []),
                "--lengths", *grid.get("lengths", spec["lengths"]), "--lrs", *grid.get("lrs", spec["lrs"]),
                "--wds", *grid.get("wds", spec["wds"]),
                *(["--set", *[f"{key}={json.dumps(value)}" for key, value in overrides.items()]] if overrides else []),
                "--tag", tag if epochs == 100 else f"{tag}-ep{epochs}", "--output", output)
            tasks += read_tasks(output)
    tasks.sort(key=lambda task: -cost(task))
    append_tasks(queue, tasks + [driver("refine", round_, queue)])
    print(f"appended {len(tasks)} search runs and the refine driver")


def refine(round_, queue):
    spec, outputs = ROUNDS[round_], []
    for metric in ("best_val_top1", "last_val_top1"):
        selection, output = files(round_, f"sel_ad_top3_{metric}.json"), files(round_, f"q_ad_seeds_{metric}.jsonl")
        run("collect.py", "--benchmark", spec.get("benchmark", "vtab"), "--runs", RUNS, "--prefix", spec["search"], "--methods", "pib",
            "--metric", metric, "--select", selection, "--top", "3")
        run("make_queue.py", "--benchmark", spec.get("benchmark", "vtab"), "--stage", "refine", "--prefix", spec["refine"], "--methods", "pib",
            *(["--tasks", *spec["tasks"]] if spec["tasks"] else []),
            "--selection", selection, "--seeds", "1", "2", "--output", output)
        outputs.append(output)
    if spec["variants"]:
        top1 = files(round_, "sel_ad_top1.json")
        run("collect.py", "--benchmark", spec.get("benchmark", "vtab"), "--runs", RUNS, "--prefix", spec["search"], "--methods", "pib",
            "--select", top1)
        for name, extra in (("ls0.3", ("--loss-scale", "0.3")), ("ls0.1", ("--loss-scale", "0.1")),
                            ("gap", ("--set", 'model.features="cls_gap"', "--tag", "gap"))):
            output = files(round_, f"q_ad_{name}.jsonl")
            run("make_queue.py", "--benchmark", spec.get("benchmark", "vtab"), "--stage", "refine", "--prefix", spec["refine"],
                "--methods", "pib", *(["--tasks", *spec["tasks"]] if spec["tasks"] else []),
                "--selection", top1, "--seeds", "0", "1", "2", *extra, "--output", output)
            outputs.append(output)
    seen, tasks = set(), []
    for output in outputs:
        for task in read_tasks(output):
            rest = task["name"].split("/", 1)[1]
            if rest not in seen and not any(Path(RUNS, prefix, rest, "result.json").exists()
                                            for prefix in (spec["search"], spec["refine"])):
                seen.add(rest)
                tasks.append(task)
    tasks.sort(key=lambda task: -cost(task))
    append_tasks(queue, tasks + [driver("final", round_, queue)])
    print(f"appended {len(tasks)} refinement runs and the final driver")


def final(round_, queue):
    spec, prefixes = ROUNDS[round_], candidates(round_)
    best, last = files(round_, "sel_ad_p800.json"), files(round_, "sel_ad_p1000.json")
    common = ("collect.py", "--benchmark", spec.get("benchmark", "vtab"), "--runs", RUNS, "--prefix", *prefixes, "--protocol", "800",
              "--methods", "pib", "--min-seeds", "3")
    print(run(*common, "--select", best, "--metric", "best_val_top1"))
    print(run(*common, "--select", last, "--metric", "last_val_top1"))
    run("eval_selected.py", "--runs", RUNS, "--selection", best, "--prefix", *prefixes, "--protocol", "800",
        "--output", files(round_, "q_ad_tests_all.jsonl"))
    tests = [task for task in read_tasks(files(round_, "q_ad_tests_all.jsonl"))
             if "test_top1" not in json.loads((Path(task["cmd"].split("--run ", 1)[1]) / "result.json").read_text())]
    run("make_queue.py", "--benchmark", spec.get("benchmark", "vtab"), "--stage", "final", "--prefix", spec["final"], "--protocol", "1000",
        "--methods", "pib", "--selection", last, "--seeds", "0", "1", "2", "3", "4",
        *(["--tasks", *spec["tasks"]] if "candidates" in spec and spec["tasks"] else []),  # rounds on their own data
        "--output", files(round_, "q_ad_p1000.jsonl"))
    finals = [task for task in read_tasks(files(round_, "q_ad_p1000.jsonl"))  # reuse identical earlier runs
              if not any(Path(RUNS, prefix, task["name"].split("/", 1)[1], "result.json").exists()
                         for prefix in spec["reuse"])]
    finals.sort(key=lambda task: -cost(task))
    p1000 = " ".join(spec["reuse"] + [spec["final"]])
    report = (f"cd scripts && (echo '# VTAB-1k: PIB + LoRA / adapter (round {round_})'; echo;"
              f" echo '## Protocol 800: train800, best-validation checkpoint (configuration by best-epoch validation)';"
              f" python collect.py --benchmark {spec.get('benchmark', 'vtab')} --runs {RUNS} --prefix {' '.join(prefixes)} --protocol 800"
              f" --methods pib --selection {best}; echo;"
              f" echo '## Protocol 1000: 1,000 images, last checkpoint, 5 seeds (configuration by last-epoch validation)';"
              f" python collect.py --benchmark {spec.get('benchmark', 'vtab')} --runs {RUNS} --prefix {p1000} --protocol 1000 --methods pib"
              f" --selection {last}) > {P}/{spec['report']}")
    name = "driver_vtab_ad_report" if round_ == "1" else f"driver_vtab_ad{round_}_report"
    append_tasks(queue, finals + tests + [{"name": name, "driver": True, "cmd": report}])
    print(f"appended {len(finals)} protocol-1000 runs, {len(tests)} test evaluations and the report driver")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("search", "refine", "final"))
    parser.add_argument("--round", default="1", choices=sorted(ROUNDS))
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    {"search": search, "refine": refine, "final": final}[args.stage](args.round, args.queue)


if __name__ == "__main__":
    main()
