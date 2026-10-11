"""Five stratified folds of the 1,000 VTAB-1k training images (train800val200) of every vtab_png task.

<task>_cv<k>.csv lists the other four folds as train and fold k as val. Test rows are left out: the
folds only serve to choose configurations of the 1,000-image protocol, and the test split is never used.
For tasks whose val200 images show other object instances than train800 (smallNORB: val200 comes from the
official test split, like the test images), <task>_cvv<k>.csv keeps train800 plus four fifths of val200 for
training and validates on the remaining fifth of val200, so validation matches the test distribution.
"""

import argparse
import csv
import random
from pathlib import Path

from benchmarks import DATA, tasks

FOLDS = 5
VAL_ONLY = ("smallnorb_azi_png", "smallnorb_ele_png")


def assign_folds(rows, folds=FOLDS, seed=0):
    """Fold index per path: each label's images are shuffled and dealt round-robin, so folds stay balanced."""
    rng, by_label = random.Random(seed), {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row["path"])
    assignment, counter = {}, 0
    for label in sorted(by_label):
        paths = sorted(by_label[label])
        rng.shuffle(paths)
        for path in paths:
            assignment[path] = counter % folds
            counter += 1
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", default=DATA + "/vtab_manifests")
    args = parser.parse_args()
    manifests = Path(args.manifests)
    for task in tasks("vtab_png"):
        with open(manifests / f"{task}_1000.csv", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
        if len(rows) != 1000 or len({row["path"] for row in rows}) != 1000:
            raise SystemExit(f"{task}: expected 1,000 distinct training images, found {len(rows)}")
        assignment = assign_folds(rows)
        for fold in range(FOLDS):
            with open(manifests / f"{task}_cv{fold}.csv", "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "label", "split"])
                writer.writerows((row["path"], row["label"], "val" if assignment[row["path"]] == fold else "train")
                                 for row in rows)
        print(f"{task}: {FOLDS} folds of {[sum(v == f for v in assignment.values()) for f in range(FOLDS)]}")
        if task in VAL_ONLY:
            with open(manifests / f"{task}_800.csv", newline="") as stream:
                rows800 = list(csv.DictReader(stream))
            train800 = [row for row in rows800 if row["split"] == "train"]
            val200 = [row for row in rows800 if row["split"] == "val"]
            if (len(train800), len(val200)) != (800, 200):
                raise SystemExit(f"{task}: expected train800 and val200, found {len(train800)} and {len(val200)}")
            held = assign_folds(val200)
            for fold in range(FOLDS):
                with open(manifests / f"{task}_cvv{fold}.csv", "w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["path", "label", "split"])
                    writer.writerows((row["path"], row["label"], "train") for row in train800)
                    writer.writerows((row["path"], row["label"], "val" if held[row["path"]] == fold else "train")
                                     for row in val200)
            print(f"{task}: val200-only folds of {[sum(v == f for v in held.values()) for f in range(FOLDS)]}")


if __name__ == "__main__":
    main()
