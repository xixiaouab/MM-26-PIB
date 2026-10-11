"""Queue driver, stage 3 of the VTAB-1k push.

Selects each task's configuration among those with at least three seeds (validation accuracy only),
queues the test evaluation of the chosen runs (train800, best-validation checkpoint) and the
1000-image protocol for the chosen configurations with five seeds (last checkpoint), and finally a
report driver that writes both summaries to $PIB_ROOT/vtab_report_final.md.
"""

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks import expected_cost
from queue_append import append_tasks

P = "/lustre/orion/csc662/proj-shared/xxiao/PIB"
PREFIXES = ["vtab_s1", "vtab_s1x", "vtab_f1", "vtab_r1", "vtab_v1", "vtab_sgd1", "vtab_b1"]


def run(*args):
    result = subprocess.run([P + "/envs/pib/bin/python", *args], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr[-2000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    selection = P + "/sel_vtab_final2.json"
    print(run("collect.py", "--benchmark", "vtab", "--prefix", *PREFIXES, "--protocol", "800",
              "--select", selection, "--min-seeds", "3"))
    run("eval_selected.py", "--selection", selection, "--prefix", *PREFIXES, "--protocol", "800",
        "--output", P + "/q_final2_tests_all.jsonl")
    tests = []
    for line in Path(P + "/q_final2_tests_all.jsonl").read_text().splitlines():
        task = json.loads(line)
        run_dir = Path(task["cmd"].split("--run ", 1)[1])
        if "test_top1" not in json.loads((run_dir / "result.json").read_text()):
            tests.append(task)
    run("make_queue.py", "--benchmark", "vtab", "--stage", "final", "--prefix", "vtab_p1000b", "--protocol", "1000",
        "--selection", selection, "--seeds", "0", "1", "2", "3", "4", "--output", P + "/q_final2_p1000.jsonl")
    finals = [json.loads(line) for line in Path(P + "/q_final2_p1000.jsonl").read_text().splitlines()]
    finals.sort(key=lambda t: -expected_cost(t["name"].split("/")[2], t["set"]["model.prompt_length"])
                * t["set"].get("training.epochs", 100))
    report = (f"cd scripts && (echo '# VTAB-1k final report'; echo; echo '## train800, best-validation checkpoint';"
              f" python collect.py --benchmark vtab --prefix {' '.join(PREFIXES)} --protocol 800 --selection {selection};"
              f" echo; echo '## 1000 training images, last checkpoint, 5 seeds';"
              f" python collect.py --benchmark vtab --prefix vtab_p1000b --protocol 1000) > {P}/vtab_report_final.md")
    append_tasks(args.queue, finals + tests + [{"name": "driver_vtab_report", "driver": True, "cmd": report}])
    print(f"appended {len(finals)} protocol-1000 runs, {len(tests)} test evaluations and the report driver")


if __name__ == "__main__":
    main()
