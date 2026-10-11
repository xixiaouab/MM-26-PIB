"""Write PIB manifests for the 19 VTAB-1k tasks from the common preprocessed release.

Each task directory holds images/ and train800.txt, val200.txt, train800val200.txt and test.txt,
one "relative/path.jpg label" per line. Two manifests are written per task:

    <task>_800.csv   train = train800, val = val200, test = test  (hyperparameter search and selection)
    <task>_1000.csv  train = train800val200, val = val200, test = test  (final training on all 1,000
                     images; val200 is part of the training data here, so it only tracks training and
                     the last checkpoint is the one evaluated)
"""

import argparse
import csv
from pathlib import Path

GROUPS = {
    "natural": ["cifar", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397"],
    "specialized": ["patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"],
    "structured": ["clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori",
                   "smallnorb_azi", "smallnorb_ele"],
}
TASKS = [task for tasks in GROUPS.values() for task in tasks]


def read_split(task_dir, name):
    rows = []
    for line in (task_dir / name).read_text().splitlines():
        if line.strip():
            path, label = line.rsplit(" ", 1)
            rows.append((path, int(label)))
    return rows


def write(path, splits):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split"])
        for split, rows in splits:
            writer.writerows((image, label, split) for image, label in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory containing the 19 task folders")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        task_dir = root / task
        train, val = read_split(task_dir, "train800.txt"), read_split(task_dir, "val200.txt")
        full, test = read_split(task_dir, "train800val200.txt"), read_split(task_dir, "test.txt")
        if (len(train), len(val), len(full)) != (800, 200, 1000):
            raise ValueError(f"{task}: unexpected split sizes {len(train)}/{len(val)}/{len(full)}")
        prefix = lambda rows: [(f"{task}/{path}", label) for path, label in rows]
        write(output / f"{task}_800.csv", [("train", prefix(train)), ("val", prefix(val)), ("test", prefix(test))])
        write(output / f"{task}_1000.csv", [("train", prefix(full)), ("val", prefix(val)), ("test", prefix(test))])
        classes = len({label for _, label in train + val + test})
        print(f"{task:22s} classes={classes:3d} test={len(test)}")


if __name__ == "__main__":
    main()
