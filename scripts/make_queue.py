"""Write a queue of runs (JSON lines) for scripts/queue_worker.py.

search  Every combination of prompt length, learning rate and weight decay for the given seeds,
        scored on the validation split only (no test evaluation).
refine  More seeds (validation only) for the configurations listed per task by collect.py --select
        --top K; with --loss-scale, the PIB regularizer weights are scaled for those configurations.
final   The configuration selected per task by collect.py --select, for several seeds, evaluated on
        the test split. VTAB-1k protocol 800 trains on train800 and tests the best-validation
        checkpoint; protocol 1000 trains on all 1,000 images and tests the last checkpoint. FGVC and
        HTA (protocol std) test the best-validation checkpoint.

--loss-scale multiplies all PIB regularizer weights of the base config (runs are tagged ls<scale>).
"""

import argparse
import itertools
import json
from pathlib import Path

import yaml

from benchmarks import BENCHMARKS, expected_cost, tasks as benchmark_tasks

VPT_LOSS = {"loss.compression": 0.0, "loss.sufficiency": 0.0, "loss.path_weight": 0.0, "loss.routing": False}


ROOT = Path(__file__).resolve().parents[1]


def scaled_loss(config_path, scale):
    path = Path(config_path)
    loss = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text())["loss"]
    scale_weights = lambda value: ({key: round(scale * float(v), 12) for key, v in value.items()} if isinstance(value, dict)
                                   else round(scale * float(value), 12))
    return {"loss.compression": scale_weights(loss["compression"]),
            "loss.sufficiency": scale_weights(loss["sufficiency"]),
            "loss.path_weight": round(scale * float(loss["path_weight"]), 12)}


def task_spec(prefix, method, task, protocol, length, lr, wd, seed, manifest, config, evaluate,
              extra=None, tag=""):
    name = f"{prefix}/{method}/{task}/L{length}_lr{lr:g}_wd{wd:g}_p{protocol}_s{seed}{'_' + tag if tag else ''}"
    settings = {"data.manifest": manifest, "model.prompt_length": length, "training.learning_rate": lr,
                "training.weight_decay": wd, "training.seed": seed}
    settings.update(VPT_LOSS if method == "vpt" else {})
    settings.update(extra or {})
    return {"name": name, "config": config, "set": settings, "eval": evaluate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--stage", choices=("search", "refine", "final"), required=True)
    parser.add_argument("--prefix", required=True, help="run directory prefix, e.g. vtab_s1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=["vpt", "pib"], choices=("vpt", "pib"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--lengths", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-3])
    parser.add_argument("--wds", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--protocol", help="vtab: 800 or 1000; others: std")
    parser.add_argument("--selection", help="selection JSON from collect.py (final stage)")
    parser.add_argument("--loss-scale", type=float, help="multiply the PIB regularizer weights")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help="extra overrides for every run")
    parser.add_argument("--tag", default="", help="label for runs with --set overrides (joined to existing tags)")
    parser.add_argument("--longest-first", action="store_true",
                        help="order runs by expected run time (training images x tokens), longest first")
    args = parser.parse_args()
    spec = BENCHMARKS[args.benchmark]
    protocol = args.protocol or spec["protocols"][0]
    if protocol not in spec["protocols"]:
        parser.error(f"{args.benchmark} protocols: {spec['protocols']}")
    names = args.tasks or benchmark_tasks(args.benchmark)
    extra = {key: json.loads(value) for key, value in (item.split("=", 1) for item in args.set)}
    tag = ""
    if args.loss_scale is not None:
        extra.update(scaled_loss(spec["config"], args.loss_scale))
        tag = f"ls{args.loss_scale:g}"
    join = lambda *tags: "-".join(t for t in tags if t)
    tag = join(tag, args.tag)
    manifest = lambda task, p: spec["manifest"].format(task=task, protocol=p)
    runs = []
    if args.stage == "search":
        methods = ["pib"] if args.loss_scale is not None else args.methods
        for method, task, length, lr, wd, seed in itertools.product(methods, names, args.lengths, args.lrs,
                                                                    args.wds, args.seeds):
            runs.append(task_spec(args.prefix, method, task, protocol, length, lr, wd, seed,
                                  manifest(task, "800" if args.benchmark == "vtab" else protocol),
                                  spec["config"], None, extra, tag))
    elif args.stage == "refine":
        selection = json.loads(Path(args.selection).read_text())
        methods = ["pib"] if args.loss_scale is not None else args.methods
        for method, task, seed in itertools.product(methods, names, args.seeds):
            configs = selection[method][task]
            for chosen in configs if isinstance(configs, list) else [configs]:
                settings = dict(chosen.get("set", {}))
                kept = chosen.get("tag", "")
                if args.loss_scale is not None:  # a new loss scale replaces any earlier one in the tag
                    kept = "-".join(t for t in kept.split("-") if t and not t.startswith("ls"))
                runs.append(task_spec(args.prefix, method, task, protocol, chosen["length"], chosen["lr"],
                                      chosen["wd"], seed, manifest(task, "800" if args.benchmark == "vtab" else protocol),
                                      spec["config"], None, {**settings, **extra}, join(kept, tag)))
    else:
        selection = json.loads(Path(args.selection).read_text())
        evaluate = "last" if protocol == "1000" else "best"
        for method, task, seed in itertools.product(args.methods, names, args.seeds):
            chosen = selection[method][task]
            runs.append(task_spec(args.prefix, method, task, protocol, chosen["length"], chosen["lr"], chosen["wd"],
                                  seed, manifest(task, protocol), spec["config"], evaluate,
                                  {**chosen.get("set", {}), **extra}, chosen.get("tag", "")))
    if args.longest_first:
        runs.sort(key=lambda run: -expected_cost(run["name"].split("/")[2], run["set"]["model.prompt_length"]))
    Path(args.output).write_text("".join(json.dumps(run) + "\n" for run in runs))
    print(f"{len(runs)} runs -> {args.output}")


if __name__ == "__main__":
    main()
