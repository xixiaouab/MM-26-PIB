import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from pib.data import Sample, apply_protocol, build_loader, read_manifest, scan_imagefolder
from pib.diagnose import diagnose, build_diagnostic_loader
from pib.evaluate import evaluate_checkpoint
from pib.predict import predict
from pib.prepare import main as prepare
from pib.train import run_training


def tiny_dataset(tmp_path):
    root = tmp_path / "images"
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        for index in range(count):
            directory = root / split / str(index % 2)
            directory.mkdir(parents=True, exist_ok=True)
            array = np.random.default_rng(index).integers(0, 256, (36, 36, 3), dtype=np.uint8)
            Image.fromarray(array).save(directory / f"{index}.png")
    manifest = tmp_path / "splits.csv"
    prepare(["--root", str(root), "--output", str(manifest)])
    return {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "prompt_length": 2,
                      "backbone_kwargs": {"img_size": 32, "patch_size": 8, "embed_dim": 24, "depth": 2, "num_heads": 3}},
            "data": {"root": str(root), "manifest": str(manifest), "image_size": 32,
                     "resize_size": 36, "num_workers": 0, "protocol": "official"},
            "training": {"epochs": 2, "warmup_epochs": 1, "batch_size": 4, "seed": 7,
                         "learning_rate": .001, "regularization_warmup": 1},
            "loss": {"normalization": "mean", "routing": True}}


def test_train_evaluate_predict_diagnose_and_resume(tmp_path):
    config = tiny_dataset(tmp_path)
    output = tmp_path / "run"
    result = run_training(config, output, device="cpu")
    assert result["epochs"] == 2
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert "objective" in checkpoint and "routing_logits" in checkpoint["objective"]
    metrics = evaluate_checkpoint(output / "best.pt", device="cpu", output=tmp_path / "test")
    assert metrics["samples"] == 4 and 0 <= metrics["top1"] <= 100
    image = next((Path(config["data"]["root"]) / "test").rglob("*.png"))
    predictions = predict(output / "best.pt", image, device="cpu")
    assert len(predictions) == 2
    assert sum(item["probability"] for item in predictions) == pytest.approx(1)
    rows = diagnose(output / "best.pt", tmp_path / "layers.csv", device="cpu")
    assert len(rows) == 2 and rows[-1]["trajectory"] == pytest.approx(1)
    config["training"]["epochs"] = 3
    resumed = run_training(config, output, resume=output / "last.pt", device="cpu")
    assert resumed["epochs"] == 3
    assert torch.load(output / "last.pt", weights_only=True)["epoch"] == 2


def test_split_integrity_class_mapping_and_test_only_loading(tmp_path):
    config = tiny_dataset(tmp_path)
    records = read_manifest(config["data"]["manifest"], config["data"]["root"])
    assert records == apply_protocol(records, "fgvc", seed=99)
    paths = [record.path for record in records]
    assert len(paths) == len(set(paths))
    test_rows = [record for record in records if record.split == "test"]
    only_test = tmp_path / "test_only.csv"
    with only_test.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split"])
        writer.writerows((str(row.path), row.label, row.split) for row in test_rows)
    config["data"]["manifest"] = str(only_test)
    assert len(build_loader(config, "test").dataset) == 4
    with only_test.open("a") as stream:
        stream.write(f"{test_rows[0].path},0,train\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(only_test)


def test_generated_vtab_split_is_seeded_and_preserves_holdout():
    training = [Sample(Path(f"/{i}.png"), i % 10, "train", str(i)) for i in range(1000)]
    holdout = Sample(Path("/test.png"), 0, "test", "test")
    result = apply_protocol(training + [holdout], "vtab", seed=3)
    assert sum(item.split == "train" for item in result) == 800
    assert sum(item.split == "val" for item in result) == 200
    assert result[-1] == holdout
    assert result == apply_protocol(training + [holdout], "vtab", seed=3)


def test_diagnostic_batches_shuffle_sorted_classes_once_and_repeatably(tmp_path):
    config = tiny_dataset(tmp_path)
    config["training"]["batch_size"] = 2
    config["data"]["diagnostic_seed"] = 42
    ordinary = list(build_loader(config, "test"))
    assert all(batch["labels"].unique().numel() == 1 for batch in ordinary)
    first = list(build_diagnostic_loader(config))
    second = list(build_diagnostic_loader(config))
    first_ids = [identifier for batch in first for identifier in batch["ids"]]
    second_ids = [identifier for batch in second for identifier in batch["ids"]]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)) == 4
    assert set(first_ids) == {identifier for batch in ordinary for identifier in batch["ids"]}
    assert any(batch["labels"].unique().numel() == 2 for batch in first)
