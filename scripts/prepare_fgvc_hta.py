"""Extract the FGVC and HTA datasets and write PIB manifests (path,label,split).

FGVC (CUB-200-2011, NABirds, Oxford Flowers, Stanford Dogs, Stanford Cars): official test split; the
official training split is divided 90/10 into train/val, stratified by class with a fixed seed
(Flowers keeps its official train/val/test).
HTA adds CIFAR-10, CIFAR-100, DTD, Food-101, GTSRB and SVHN with the DAM-VP protocol: 80/20
train/val from the official training split (CIFAR and SVHN in file order, GTSRB and Food-101 after
a seeded shuffle), DTD partition 1, official test splits. HTA reuses the FGVC splits of CUB,
NABirds, Stanford Dogs and Oxford Flowers.

    python scripts/prepare_fgvc_hta.py --raw data/raw --images data/images --manifests data/manifests [names]
"""

import argparse
import csv
import json
import pickle
import random
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42


def extract(archive, destination):
    marker = destination / f".extracted_{archive.name}"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    else:
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    marker.touch()


def stratified_val(items, fraction, seed=SEED):
    """Split (path, label) items into train/val with `fraction` of each class in val."""
    rng = random.Random(seed)
    by_class = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)
    train, val = [], []
    for label in sorted(by_class):
        group = sorted(by_class[label])
        rng.shuffle(group)
        count = int(round(len(group) * fraction))
        val += group[:count]
        train += group[count:]
    return train, val


def ordered_val(items, fraction):
    cut = int((1 - fraction) * len(items))
    return items[:cut], items[cut:]


def shuffled_val(items, fraction, seed=SEED):
    items = list(items)
    random.Random(seed).shuffle(items)
    return ordered_val(items, fraction)


def write_manifest(path, root, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split"])
        for split, items in splits.items():
            counts[split] = len(items)
            for file, label in items:
                writer.writerow([Path(file).relative_to(root).as_posix(), label, split])
    labels = {label for items in splits.values() for _, label in items}
    print(f"{path.stem:10s} classes={len(labels)} " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)


def save_png(args):
    array, path = args
    if not path.exists():
        Image.fromarray(array).save(path)


def export_arrays(arrays, labels, directory, workers):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{index:06d}.png" for index in range(len(arrays))]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(save_png, zip(arrays, paths), chunksize=512))
    return [(str(path), int(label)) for path, label in zip(paths, labels)]


def cub(raw, images):
    extract(raw / "CUB_200_2011.tgz", images / "cub")
    base = next((images / "cub").rglob("images.txt")).parent
    files = dict(line.split() for line in (base / "images.txt").read_text().splitlines())
    labels = {k: int(v) - 1 for k, v in (line.split() for line in (base / "image_class_labels.txt").read_text().splitlines())}
    is_train = {k: v == "1" for k, v in (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
    items = {k: (str(base / "images" / files[k]), labels[k]) for k in files}
    train, val = stratified_val([items[k] for k in files if is_train[k]], 0.1)
    return {"train": train, "val": val, "test": [items[k] for k in files if not is_train[k]]}


def nabirds(raw, images):
    extract(raw / "nabirds.zip", images / "nabirds")
    base = next((images / "nabirds").rglob("train_test_split.txt")).parent
    files = dict(line.split() for line in (base / "images.txt").read_text().splitlines())
    raw_labels = {k: int(v) for k, v in (line.split() for line in (base / "image_class_labels.txt").read_text().splitlines())}
    index = {label: i for i, label in enumerate(sorted(set(raw_labels.values())))}
    is_train = {k: v == "1" for k, v in (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
    items = {k: (str(base / "images" / files[k]), index[raw_labels[k]]) for k in files}
    train, val = stratified_val([items[k] for k in files if is_train[k]], 0.1)
    return {"train": train, "val": val, "test": [items[k] for k in files if not is_train[k]]}


def flowers(raw, images):
    from scipy.io import loadmat

    extract(raw / "102flowers.tgz", images / "flowers")
    labels = loadmat(raw / "imagelabels.mat")["labels"][0] - 1
    ids = loadmat(raw / "setid.mat")
    path = lambda i: str(images / "flowers" / "jpg" / f"image_{i:05d}.jpg")
    split = lambda key: [(path(int(i)), int(labels[int(i) - 1])) for i in ids[key][0]]
    return {"train": split("trnid"), "val": split("valid"), "test": split("tstid")}


def dogs(raw, images):
    from scipy.io import loadmat

    extract(raw / "dogs_images.tar", images / "dogs")
    extract(raw / "dogs_lists.tar", images / "dogs")
    def read(name):
        mat = loadmat(images / "dogs" / name)
        return [(str(images / "dogs" / "Images" / f[0][0]), int(l[0]) - 1) for f, l in zip(mat["file_list"], mat["labels"])]
    train, val = stratified_val(read("train_list.mat"), 0.1)
    return {"train": train, "val": val, "test": read("test_list.mat")}


def cars(raw, images):
    """Stanford Cars from the Hugging Face copy tanganke/stanford_cars (stanford_cars_{train,test}-*.parquet).

    The image files of the fast.ai archive are not in the order of its cars_annos.mat, so the labels
    come from the parquet copy, whose per-class image counts must equal those of the official
    cars_annos.mat for both splits.
    """
    from collections import Counter

    import pyarrow.parquet as pq
    from scipy.io import loadmat

    splits = {}
    for split in ("train", "test"):
        items, directory = [], images / "cars_hf" / split
        directory.mkdir(parents=True, exist_ok=True)
        for part, file in enumerate(sorted(raw.glob(f"stanford_cars_{split}-*.parquet"))):
            table = pq.read_table(file, columns=["image", "label"]).to_pydict()
            for index, (image, label) in enumerate(zip(table["image"], table["label"])):
                data = image["bytes"]
                path = directory / f"{part}_{index:05d}{'.jpg' if data[:2] == bytes([0xFF, 0xD8]) else '.png'}"
                if not path.exists():
                    path.write_bytes(data)
                items.append((str(path), int(label)))
        splits[split] = items
    extract(raw / "stanford-cars.tgz", images / "cars")
    annotations = loadmat(next((images / "cars").rglob("cars_annos.mat")))["annotations"][0]
    for split, flag in (("train", 0), ("test", 1)):
        official = Counter(int(a["class"][0][0]) - 1 for a in annotations if int(a["test"][0][0]) == flag)
        if Counter(label for _, label in splits[split]) != official:
            raise ValueError(f"{split}: per-class counts differ from the official cars_annos.mat")
    train, val = stratified_val(splits["train"], 0.1)
    return {"train": train, "val": val, "test": splits["test"]}


def dtd(raw, images):
    extract(raw / "dtd-r1.0.1.tar.gz", images / "dtd")
    base = images / "dtd" / "dtd"
    classes = sorted(p.name for p in (base / "images").iterdir() if p.is_dir())
    index = {name: i for i, name in enumerate(classes)}
    read = lambda name: [(str(base / "images" / line), index[line.split("/")[0]])
                         for line in (base / "labels" / name).read_text().split()]
    return {"train": read("train1.txt"), "val": read("val1.txt"), "test": read("test1.txt")}


def cifar(raw, images, which, workers):
    extract(raw / f"cifar-{which}-python.tar.gz", images / f"cifar{which}")
    base = next(p for p in (images / f"cifar{which}").glob("cifar-*") if p.is_dir())
    key = b"labels" if which == "10" else b"fine_labels"
    def load(names):
        data, labels = [], []
        for name in names:
            with open(base / name, "rb") as handle:
                batch = pickle.load(handle, encoding="bytes")
            data.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels += batch[key]
        return np.concatenate(data), labels
    train_names = [f"data_batch_{i}" for i in range(1, 6)] if which == "10" else ["train"]
    test_names = ["test_batch"] if which == "10" else ["test"]
    train = export_arrays(*load(train_names), images / f"cifar{which}" / "png" / "train", workers)
    test = export_arrays(*load(test_names), images / f"cifar{which}" / "png" / "test", workers)
    train, val = ordered_val(train, 0.2)
    return {"train": train, "val": val, "test": test}


def svhn(raw, images, workers):
    from scipy.io import loadmat

    def load(name, split):
        mat = loadmat(raw / name)
        labels = mat["y"][:, 0] % 10  # label 10 is digit 0
        return export_arrays(mat["X"].transpose(3, 0, 1, 2), labels, images / "svhn" / split, workers)
    train, val = ordered_val(load("svhn_train_32x32.mat", "train"), 0.2)
    return {"train": train, "val": val, "test": load("svhn_test_32x32.mat", "test")}


def gtsrb(raw, images):
    for name in ("GTSRB-Training_fixed.zip", "GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_GT.zip"):
        extract(raw / name, images / "gtsrb")
    root = images / "gtsrb"
    training = next(p for p in root.rglob("Training") if p.is_dir())
    items = [(str(f), int(f.parent.name)) for f in sorted(training.rglob("*.ppm"))]
    train, val = shuffled_val(items, 0.2)
    test_dir = next(p for p in root.rglob("Images") if p.is_dir() and "Test" in str(p))
    rows = next(root.rglob("GT-final_test.csv")).read_text().splitlines()[1:]
    test = [(str(test_dir / r.split(";")[0]), int(r.split(";")[-1])) for r in rows]
    return {"train": train, "val": val, "test": test}


def food101(raw, images):
    extract(raw / "food-101.tar.gz", images / "food101")
    base = images / "food101" / "food-101"
    classes = sorted(json.loads((base / "meta" / "train.json").read_text()))
    index = {name: i for i, name in enumerate(classes)}
    read = lambda split: [(str(base / "images" / f"{f}.jpg"), index[c])
                          for c, files in json.loads((base / "meta" / f"{split}.json").read_text()).items() for f in files]
    train, val = shuffled_val(read("train"), 0.2)
    return {"train": train, "val": val, "test": read("test")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("names", nargs="*", default=["flowers", "dogs", "cub", "nabirds", "cars", "dtd",
                                                     "cifar10", "cifar100", "svhn", "gtsrb", "food101"])
    args = parser.parse_args()
    raw, images, manifests = Path(args.raw), Path(args.images), Path(args.manifests)
    builders = {"cub": lambda: cub(raw, images), "nabirds": lambda: nabirds(raw, images),
                "flowers": lambda: flowers(raw, images), "dogs": lambda: dogs(raw, images),
                "cars": lambda: cars(raw, images), "dtd": lambda: dtd(raw, images),
                "cifar10": lambda: cifar(raw, images, "10", args.workers),
                "cifar100": lambda: cifar(raw, images, "100", args.workers),
                "svhn": lambda: svhn(raw, images, args.workers), "gtsrb": lambda: gtsrb(raw, images),
                "food101": lambda: food101(raw, images)}
    for name in args.names:
        write_manifest(manifests / f"{name}.csv", images, builders[name]())


if __name__ == "__main__":
    main()
