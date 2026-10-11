"""Lossless copies of VTAB-1k tasks whose common release re-encodes small images as JPEG.

The common VTAB-1k release stores, e.g., CIFAR-100 and SVHN as 32 x 32 JPEG files, while TFDS (used by
VPT and VFPT) serves the original pixels. Each stored image of the train800 / val200 / train800val200 /
test lists is matched to the original image with the same label and the smallest mean absolute pixel
difference, searching the official split the list was drawn from (TFDS shuffles examples when it builds a
dataset, so the VTAB subsets cannot be recovered by index). A match is trusted when it is close, clearly
better than the runner-up, and its original is used once in the list; trusted images are written with the
original pixels and the rest keep the decoded JPEG pixels, so the lists, order and labels are unchanged.
The copy is abandoned if more than 2% of a list is untrusted. The manifests <task>_png_800.csv and
<task>_png_1000.csv mirror <task>_800.csv and <task>_1000.csv row by row. smallNORB follows the VTAB
splits: train800 from the official training set, val200 and test from the official test set (camera 0,
azimuth label = azimuth / 2), matched in grayscale.
"""

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

P = "/lustre/orion/csc662/proj-shared/xxiao/PIB"


def match(queries, query_labels, originals, original_labels):
    """Index of the closest same-label original for each query, its distance and the runner-up distance."""
    queries, originals = queries.reshape(len(queries), -1).astype(np.float32), originals.reshape(len(originals), -1)
    by_label = {label: np.flatnonzero(original_labels == label) for label in np.unique(original_labels)}
    index = np.empty(len(queries), dtype=np.int64)
    best, second = np.empty(len(queries)), np.empty(len(queries))
    for i, (query, label) in enumerate(zip(queries, query_labels)):
        pool = by_label[label]
        distances = np.abs(originals[pool].astype(np.float32) - query).mean(1)
        order = np.argsort(distances)[:2]
        index[i], best[i] = pool[order[0]], distances[order[0]]
        second[i] = distances[order[1]] if len(order) > 1 else np.inf
    return index, best, second


def trusted(index, best, second, max_distance=20.0, min_ratio=2.0, min_gap=4.0):
    """Matches that are close, clearly better than the runner-up, and whose original is used only once."""
    keep = (best <= max_distance) & (second >= np.maximum(min_ratio * best, best + min_gap))
    values, counts = np.unique(index, return_counts=True)
    return keep & ~np.isin(index, values[counts > 1])


def load_cifar(data):
    def load(path):
        with open(path, "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        return batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(batch[b"fine_labels"])
    base = data / "images/cifar100/cifar-100-python"
    return {"train": load(base / "train"), "test": load(base / "test")}


def load_svhn(data):
    from scipy.io import loadmat

    def load(path):
        mat = loadmat(path)
        return np.ascontiguousarray(mat["X"].transpose(3, 0, 1, 2)), mat["y"][:, 0] % 10  # label 10 is digit 0
    return {"train": load(data / "raw/svhn_train_32x32.mat"), "test": load(data / "raw/svhn_test_32x32.mat")}


def read_norb(path):
    """A matrix in the smallNORB binary format (gzip-compressed)."""
    with gzip.open(path) as handle:
        magic, ndim = np.frombuffer(handle.read(8), dtype="<i4")
        dims = np.frombuffer(handle.read(4 * max(3, int(ndim))), dtype="<i4")[:ndim]
        dtype = {0x1E3D4C55: np.uint8, 0x1E3D4C54: np.int32}[int(magic)]
        return np.frombuffer(handle.read(), dtype=dtype).reshape(dims)


def load_smallnorb(data, attribute):
    def load(split, prefix):
        base = data / "raw/smallnorb" / f"smallnorb-{prefix}-{split}"
        images, info = read_norb(f"{base}-dat.mat.gz")[:, 0], read_norb(f"{base}-info.mat.gz")
        return np.ascontiguousarray(images), info[:, 2] // 2 if attribute == "azimuth" else info[:, 1]
    return {"train": load("training", "5x46789x9x18x6x2x96x96"), "test": load("testing", "5x01235x9x18x6x2x96x96")}


TRAIN_LISTS = {"test": ("test",)}  # every other list comes from the official training split
NORB_LISTS = {"train800": ("train",), "val200": ("test",), "train800val200": ("train", "test"), "test": ("test",)}
# task -> (loader, image mode, list sources, minimum distance gap to the runner-up). The gap only guards
# against exact duplicates; the runner-up must also be twice as far. smallNORB images are smooth grayscale
# photographs whose JPEG error is about 1 (CIFAR and SVHN: about 6), and the same pose under another
# lighting is a close runner-up, so its absolute gap is smaller.
TASKS = {"cifar": (load_cifar, "RGB", TRAIN_LISTS, 4.0), "svhn": (load_svhn, "RGB", TRAIN_LISTS, 4.0),
         "smallnorb_azi": (lambda data: load_smallnorb(data, "azimuth"), "L", NORB_LISTS, 0.5),
         "smallnorb_ele": (lambda data: load_smallnorb(data, "elevation"), "L", NORB_LISTS, 0.5)}


def read_rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def build(task, originals, data, mode="RGB", sources=TRAIN_LISTS, min_gap=4.0):
    """Write the lossless copy of one task and its manifests; returns the per-list matching summary.

    sources maps a list directory (train800, val200, train800val200, test) to the official splits its
    images were drawn from; lists not named come from the training split.
    """
    root, manifests = data / "vtab-1k", data / "vtab_manifests"
    tables = {protocol: read_rows(manifests / f"{task}_{protocol}.csv") for protocol in ("800", "1000")}
    lists = {}  # list directory -> {path: label}, each stored image once
    for rows in tables.values():
        for row in rows:
            lists.setdefault(Path(row["path"]).parent.name, {})[row["path"]] = int(row["label"])
    mapping, summary = {}, {}
    for name, items in sorted(lists.items()):
        paths, source = list(items), "+".join(sources.get(name, ("train",)))
        queries = np.stack([np.asarray(Image.open(root / path).convert(mode)) for path in paths])
        images = np.concatenate([originals[split][0] for split in sources.get(name, ("train",))])
        labels = np.concatenate([originals[split][1] for split in sources.get(name, ("train",))])
        if queries.shape[1:] != images.shape[1:]:
            raise SystemExit(f"{task}/{name}: stored images {queries.shape[1:]} differ from originals {images.shape[1:]}")
        index, best, second = match(queries, np.array([items[p] for p in paths]), images, labels)
        keep = trusted(index, best, second, min_gap=min_gap)
        ratio = second / np.maximum(best, 1e-6)
        summary[name] = {"images": len(paths), "source": source, "trusted": int(keep.sum()),
                         "mean_distance": float(best[keep].mean()) if keep.any() else None,
                         "max_trusted_distance": float(best[keep].max()) if keep.any() else None,
                         "distance_p50_p99": [round(float(q), 2) for q in np.percentile(best, [50, 99])],
                         "runner_up_ratio_p1_p50": [round(float(q), 2) for q in np.percentile(ratio, [1, 50])],
                         "untrusted_distances": [round(float(d), 2) for d in best[~keep]][:50]}
        print(json.dumps({name: summary[name]}), flush=True)
        if (~keep).mean() > 0.02:
            raise SystemExit(f"{task}/{name}: {int((~keep).sum())} of {len(paths)} matches are not trusted")
        for path, i, query, ok in zip(paths, index, queries, keep):
            target = Path(path.replace(f"{task}/", f"{task}_png/", 1)).with_suffix(".png")
            (root / target).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[i] if ok else query).save(root / target)
            mapping[path] = target.as_posix()
    for protocol, rows in tables.items():
        with open(manifests / f"{task}_png_{protocol}.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "split"])
            writer.writerows((mapping[row["path"]], row["label"], row["split"]) for row in rows)
    (root / f"{task}_png" / "match_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {task}_png_800.csv and {task}_png_1000.csv")
    return summary


# Tasks whose stored images are byte-identical to another task's (VTAB builds both smallNORB tasks from the
# same examples); their lossless copy reuses that task's images with their own labels.
SAME_IMAGES = {"smallnorb_ele": "smallnorb_azi"}


def build_alias(task, source, data):
    """Manifests for `task` pointing at the lossless images of `source`, after checking the stored images match."""
    root, manifests = data / "vtab-1k", data / "vtab_manifests"
    checked, mapping = set(), {}
    for protocol in ("800", "1000"):
        for row in read_rows(manifests / f"{task}_{protocol}.csv"):
            path = row["path"]
            if path in checked:
                continue
            twin = path.replace(f"{task}/", f"{source}/", 1)
            target = Path(twin.replace(f"{source}/", f"{source}_png/", 1)).with_suffix(".png")
            if (root / path).read_bytes() != (root / twin).read_bytes() or not (root / target).is_file():
                raise SystemExit(f"{path} does not match {twin} or {target} is missing")
            checked.add(path)
            mapping[path] = target.as_posix()
    for protocol in ("800", "1000"):
        rows = read_rows(manifests / f"{task}_{protocol}.csv")
        with open(manifests / f"{task}_png_{protocol}.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "split"])
            writer.writerows((mapping[row["path"]], row["label"], row["split"]) for row in rows)
    print(f"wrote {task}_png manifests over {len(checked)} images identical to {source}'s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--data", default=P + "/data")
    args = parser.parse_args()
    data = Path(args.data)
    if args.task in SAME_IMAGES:
        build_alias(args.task, SAME_IMAGES[args.task], data)
        return
    loader, mode, sources, min_gap = TASKS[args.task]
    build(args.task, loader(data), data, mode, sources, min_gap)


if __name__ == "__main__":
    main()
