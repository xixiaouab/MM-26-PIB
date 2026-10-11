"""Work through a queue of experiments on one GPU.

Many workers (one per GPU, any number of jobs) share a queue file. A worker claims a task by
creating <runs>/<name>/claim exclusively, runs it in a subprocess, and moves on; finished tasks have
result.json. A worker stops taking tasks when fewer than --reserve-minutes remain before --deadline; a task that
would run past the deadline checkpoints and exits (code 75), and its claim is released so that a
later job resumes it from last.pt.

A task with "driver": true is a barrier: it runs only after every task listed before it has
finished (or failed), and its command may append further tasks to the queue file. While a driver
is pending, workers with nothing to claim wait instead of exiting, so one allocation can run
several dependent stages.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def claim(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(run_dir / "claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(handle, f"{os.environ.get('SLURM_JOB_ID', 'local')} {socket.gethostname()} {os.getpid()}\n".encode())
    os.close(handle)
    return True


def finished(runs, task):
    return (runs / task["name"] / "result.json").exists() or (runs / task["name"] / "failed").exists()


def read_queue(path):
    """Tasks in the queue file; a malformed line (e.g., a partial append) is skipped instead of stopping workers."""
    tasks = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"skipping malformed queue line {number}: {error}", file=sys.stderr, flush=True)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True, help="JSON lines, one task per line")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--deadline", type=float, default=float("inf"), help="UNIX time the job ends")
    parser.add_argument("--reserve-minutes", type=float, default=20)
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    runs = Path(args.runs)
    rank = os.environ.get("SLURM_PROCID", "0")
    done = 0
    while True:
        if time.time() > args.deadline - 60 * args.reserve_minutes:
            print(f"[worker {rank}] stopping: deadline near after {done} tasks", flush=True)
            return
        tasks = read_queue(args.queue)
        task = next((t for t in tasks if not (runs / t["name"] / "result.json").exists()
                     and not (runs / t["name"] / "claim").exists() and claim(runs / t["name"])), None)
        if task is None:
            if any(t.get("driver") and not finished(runs, t) for t in tasks):
                time.sleep(args.poll_seconds)  # a driver may still append tasks
                continue
            print(f"[worker {rank}] queue empty after {done} tasks", flush=True)
            return
        run_dir = runs / task["name"]
        if task.get("driver"):  # barrier: every task listed before the driver must be finished
            while True:
                tasks = read_queue(args.queue)
                position = next(i for i, t in enumerate(tasks) if t["name"] == task["name"])
                if all(finished(runs, t) for t in tasks[:position]):
                    break
                if time.time() > args.deadline - 60 * args.reserve_minutes:
                    (run_dir / "claim").unlink()  # let a later job run the driver
                    print(f"[worker {rank}] released driver {task['name']}: deadline near", flush=True)
                    return
                time.sleep(args.poll_seconds)
        start = time.time()
        # A task with "cmd" runs that shell command (diagnostics, drivers); others go through run_task.py.
        command = task["cmd"] if "cmd" in task else [sys.executable, "scripts/run_task.py", "--task",
                                                      json.dumps(task), "--output", str(run_dir)]
        environment = {**os.environ, "PIB_DEADLINE": str(args.deadline)} if args.deadline < float("inf") else None
        # Training checkpoints itself before the deadline; a command cannot, so it is stopped a minute before
        # the deadline and its claim released for a later job to rerun it (instead of a claim nobody holds).
        timeout = max(1.0, args.deadline - time.time() - 60) if "cmd" in task and args.deadline < float("inf") else None
        with open(run_dir / "log.txt", "a") as log:
            process = subprocess.Popen(command, shell="cmd" in task, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                (run_dir / "claim").unlink()
                print(f"[worker {rank}] {task['name']} stopped at the deadline, will rerun", flush=True)
                return
        if code == 75:  # preempted at the deadline: release the task so a later job resumes it
            (run_dir / "claim").unlink()
            print(f"[worker {rank}] {task['name']} preempted, will resume", flush=True)
            return
        if code != 0:
            (run_dir / "failed").write_text(f"exit {code}\n")
        elif "cmd" in task:
            (run_dir / "result.json").write_text(json.dumps({"name": task["name"], "cmd": task["cmd"]}) + "\n")
        done += 1
        print(f"[worker {rank}] {task['name']} exit={code} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
