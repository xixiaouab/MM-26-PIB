import csv
import io
import gzip
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import timm
import torch
from torch.nn import functional as F
import yaml

from pib.data import ImageDataset, ImageTransform, read_manifest, transform_from_config
from pib.losses import PIBObjective, layer_sufficiency, scatter_statistics
from pib.model import PromptedVisionTransformer, create_model, load_model_state, model_state
from pib.train import run_training

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vtab_lossless  # noqa: E402

TINY = {"img_size": 32, "patch_size": 8, "embed_dim": 24, "depth": 2, "num_heads": 3}


def tiny_vit():
    return timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY)


def tiny_model(seed, **options):
    torch.manual_seed(seed)
    return PromptedVisionTransformer(tiny_vit(), 3, prompt_length=2, **options)


def tiny_swin():
    return timm.create_model("swin_tiny_patch4_window7_224", num_classes=0, img_size=32, patch_size=2, window_size=4,
                             embed_dim=12, depths=(1, 1), num_heads=(1, 2))


def write_images(tmp_path, count=12):
    rows = []
    for index in range(count):
        path = tmp_path / "images" / f"{index}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (36, 36, 3), dtype=np.uint8)).save(path)
        split = "train" if index < 6 else "val" if index < 9 else "test"
        rows.append((f"images/{index}.png", index % 3, split))
    manifest = tmp_path / "splits.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)
    return manifest


def copy_into(target, source):
    target.backbone.load_state_dict(source.backbone.state_dict())
    target.prompts.load_state_dict(source.prompts.state_dict())
    target.head.load_state_dict(source.head.state_dict())


def trained_names(model):
    return {name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "vtab").glob("*.yaml")), ids=lambda path: path.stem)
def test_vtab_configs_build(path):
    config = yaml.safe_load(path.read_text())
    assert config["data"]["manifest"] == f"data/vtab_manifests/{path.stem}_1000.csv"
    options = {**config["model"], "backbone": "vit_tiny_patch16_224", "pretrained": False, "backbone_kwargs": TINY}
    model = create_model({"model": options})
    objective = PIBObjective(len(model.prompt_layers), **config["loss"])
    transform = transform_from_config({**config["data"], "image_size": 32, "resize_size": 32}, training=True)
    view = transform.normalize(transform(Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))))
    output = model(view[None].repeat(4, 1, 1, 1), return_states=True)
    assert output["logits"].shape == (4, config["model"]["num_classes"])
    labels = torch.arange(4) % config["model"]["num_classes"]
    assert torch.isfinite(objective(output, labels)["loss"])


def test_vtab_transform_is_a_plain_resize():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8))
    transform = ImageTransform(True, image_size=32, resize_size=32, augment="none", resize_mode="square")
    assert transform.deterministic and transform.horizontal_flip == 0
    first, second = transform(image), transform(image)
    assert first.size == (32, 32) and np.array_equal(np.asarray(first), np.asarray(second))
    with pytest.raises(ValueError):
        ImageTransform(True, augment="mixup")


def test_flip_augmentation_keeps_the_evaluation_geometry():
    image = Image.fromarray(np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3))
    flip = ImageTransform(True, image_size=32, resize_size=32, augment="flip", resize_mode="square", horizontal_flip=1.0)
    plain = ImageTransform(False, image_size=32, resize_size=32, augment="flip", resize_mode="square")
    assert not flip.deterministic and plain.deterministic
    assert np.array_equal(np.asarray(flip(image)), np.asarray(plain(image))[:, ::-1])


def test_dihedral_augmentation_keeps_size_and_only_permutes_pixels():
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8))
    transform = ImageTransform(training=True, image_size=32, resize_size=32, augment="d4", resize_mode="square")
    torch.manual_seed(0)
    seen = set()
    for _ in range(40):
        out = np.asarray(transform(image))
        assert out.shape == (32, 32, 3) and sorted(out.ravel().tolist()) == sorted(np.asarray(image).ravel().tolist())
        seen.add(out.tobytes())
    assert len(seen) == 8
    evaluation = ImageTransform(training=False, image_size=32, resize_size=32, augment="d4", resize_mode="square")
    np.testing.assert_array_equal(np.asarray(evaluation(image)), np.asarray(image))


def test_cached_dataset_matches_uncached_and_manifest_skips_file_checks(tmp_path):
    manifest = write_images(tmp_path)
    records = read_manifest(manifest, tmp_path, verify_files=False)
    transform = ImageTransform(True, image_size=32, resize_size=32, augment="none", resize_mode="square")
    cached, plain = ImageDataset(records, transform, cache=True), ImageDataset(records, transform)
    for index in (0, 0, 5):
        assert torch.equal(cached[index]["images"], plain[index]["images"])
    assert len(cached.cache) == 2
    missing = tmp_path / "missing.csv"
    missing.write_text("path,label,split\nnot/there.png,0,train\n")
    assert len(read_manifest(missing, tmp_path, verify_files=False)) == 1
    with pytest.raises(FileNotFoundError):
        read_manifest(missing, tmp_path)


def test_checkpoint_without_backbone_restores_trained_parameters():
    trained, fresh = tiny_model(0), tiny_model(1)
    fresh.backbone.load_state_dict(trained.backbone.state_dict())
    state = model_state(trained, include_backbone=False)
    assert state and not any(key.startswith("backbone.") for key in state)
    load_model_state(fresh, state, backbone_in_checkpoint=False)
    for key, value in trained.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key])
    with pytest.raises(ValueError):
        load_model_state(fresh, {**state, "unexpected.weight": torch.zeros(1)}, backbone_in_checkpoint=False)


def reference_scatter(features, labels):
    features = features.float()
    centers, within = [], []
    for label in labels.unique(sorted=True):
        members = features[labels == label]
        center = members.mean(0)
        centers.append(center)
        if members.shape[0] >= 2:
            within.append((members - center).square().sum())
    zero = features.sum() * 0
    intra = torch.stack(within).mean() if within else zero
    inter = (torch.stack(centers)[:, None] - torch.stack(centers)[None]).square().sum() if len(centers) >= 2 else zero
    return intra, inter, len(within), len(centers)


@pytest.mark.parametrize("labels", [[0, 0, 1, 1, 2, 5, 5, 5], [3, 3, 3, 3, 3, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7],
                                    [7, 0, 7, 1, 1, 9, 0, 2]])
def test_vectorized_sufficiency_matches_the_per_class_loop(labels):
    labels = torch.tensor(labels)
    features = torch.randn(3, 8, 5, requires_grad=True)
    expected = []
    for layer in range(3):
        intra, inter, valid, classes = reference_scatter(features[layer], labels)
        got = scatter_statistics(features[layer], labels)
        torch.testing.assert_close(got[0], intra)
        torch.testing.assert_close(got[1], inter)
        assert got[2:] == (valid, classes)
        expected.append(intra / (inter + 1e-8) if valid and classes >= 2 else features.sum() * 0)
    expected = torch.stack(expected)
    actual = layer_sufficiency(features, labels)
    torch.testing.assert_close(actual, expected)
    if expected.requires_grad and expected.abs().sum() > 0:
        grad_actual = torch.autograd.grad(actual.sum(), features, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected.sum(), features)[0]
        torch.testing.assert_close(grad_actual, grad_expected)


def test_inactive_objective_is_plain_cross_entropy():
    objective = PIBObjective(2, compression=0.0, sufficiency=0.0, path_weight=0.0, routing=False)
    assert not objective.active and PIBObjective(2).active
    logits, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
    values = objective({"logits": logits, "states": []}, labels)
    torch.testing.assert_close(values["loss"], F.cross_entropy(logits, labels))
    torch.testing.assert_close(objective(logits, labels)["loss"], values["loss"])


def test_label_smoothing_applies_to_the_task_loss_only():
    logits, labels = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 0, 1])
    off = {"compression": {"start": 0, "end": 0}, "sufficiency": {"start": 0, "end": 0}, "path_weight": 0}
    smoothed = PIBObjective(2, label_smoothing=0.1, **off)
    torch.testing.assert_close(smoothed(logits, labels)["loss"], F.cross_entropy(logits, labels, label_smoothing=0.1))
    torch.testing.assert_close(PIBObjective(2, **off)(logits, labels)["loss"], F.cross_entropy(logits, labels))
    with pytest.raises(ValueError):
        PIBObjective(2, label_smoothing=1.0)


def test_cls_gap_features_train_only_new_parameters():
    model = tiny_model(0, features="cls_gap")
    images = torch.randn(4, 3, 32, 32)
    output = model(images, return_states=True)
    assert output["logits"].shape == (4, 3) and len(output["states"]) == 2 and model.head.in_features == 48
    output["logits"].sum().backward()
    assert not any(name.startswith("backbone.") for name in trained_names(model))
    clone = tiny_model(1, features="cls_gap")
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, model_state(model, include_backbone=False), backbone_in_checkpoint=False)
    model.eval(), clone.eval()
    torch.testing.assert_close(clone(images), model(images))


def test_lora_and_adapter_start_as_the_plain_model_and_train_outside_the_backbone():
    options = {"lora": {"rank": 2, "targets": ["qkv", "fc1"]}, "adapter": {"dim": 4, "scale": 0.5}}
    model, plain = tiny_model(0, **options), tiny_model(1)
    copy_into(plain, model)
    images = torch.randn(4, 3, 32, 32)
    model.eval(), plain.eval()
    torch.testing.assert_close(model(images), plain(images))
    model.train()
    output = model(images, return_states=True)
    assert len(output["states"]) == 2
    output["logits"].sum().backward()
    trained = trained_names(model)
    assert not any(name.startswith("backbone.") for name in trained)
    assert {"lora.1_qkv.up.weight", "lora.2_fc1.up.weight", "adapters.1.up.weight"} <= trained
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("up.weight"):
                parameter.normal_()
    model.eval()
    assert not torch.allclose(model(images), plain(images))
    state = model_state(model, include_backbone=False)
    assert any(key.startswith("lora.") for key in state) and any(key.startswith("adapters.") for key in state)
    clone = tiny_model(2, **options)
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, state, backbone_in_checkpoint=False)
    clone.eval()
    torch.testing.assert_close(clone(images), model(images))
    with pytest.raises(ValueError):
        tiny_model(0, lora={"rank": 2, "targets": ["unknown"]})


def test_ssf_starts_as_the_plain_model_and_trains_only_scale_and_shift():
    model, plain = tiny_model(0, ssf=True), tiny_model(1)
    copy_into(plain, model)
    images = torch.randn(4, 3, 32, 32)
    model.eval(), plain.eval()
    torch.testing.assert_close(model(images), plain(images))
    assert len(model.ssf) == 2 + 6 * len(model.backbone.blocks) and model.ssf["1_qkv"].scale.numel() == 3 * 24
    model.train()
    model(images, return_states=True)["logits"].sum().backward()
    trained = trained_names(model)
    assert not any(name.startswith("backbone.") for name in trained)
    assert {"ssf.patch_embed.scale", "ssf.1_fc1.shift", "ssf.norm.scale"} <= trained
    with torch.no_grad():
        model.ssf["2_fc2"].scale.mul_(1.5)
    model.eval()
    assert not torch.allclose(model(images), plain(images))
    clone = tiny_model(2, ssf=True)
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, model_state(model, include_backbone=False), backbone_in_checkpoint=False)
    clone.eval()
    torch.testing.assert_close(clone(images), model(images))


@pytest.mark.parametrize("options", [{"features": "cls_gap"}, {"lora": {"rank": 2}}, {"adapter": {"dim": 4}},
                                     {"ssf": True}])
def test_swin_rejects_vit_only_options(options):
    with pytest.raises(ValueError):
        PromptedVisionTransformer(tiny_swin(), 3, prompt_length=2, **options)


def test_sgd_and_xavier_prompts(tmp_path):
    model = PromptedVisionTransformer(tiny_vit(), 3, prompt_length=4, prompt_init="xavier")
    bound = math.sqrt(6.0 / (3 * 8 * 8 + 24))
    assert all(float(p.abs().max()) <= bound for p in model.prompts.values())
    manifest = write_images(tmp_path)
    config = {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "prompt_length": 2,
                        "backbone_kwargs": TINY, "prompt_init": "xavier"},
              "data": {"root": str(tmp_path), "manifest": str(manifest), "image_size": 32, "resize_size": 32,
                       "augment": "none", "resize_mode": "square", "num_workers": 0, "verify_files": False},
              "training": {"epochs": 2, "warmup_epochs": 1, "batch_size": 6, "seed": 0, "learning_rate": 0.25,
                           "optimizer": "sgd", "weight_decay": 0.001}}
    assert run_training(config, tmp_path / "sgd", device="cpu")["epochs"] == 2
    state = torch.load(tmp_path / "sgd" / "last.pt", weights_only=True)
    assert "momentum_buffer" in str(state["optimizer"]["state"])


def smooth_images(rng, count):
    return np.stack([np.asarray(Image.fromarray(c).resize((32, 32), Image.BILINEAR))
                     for c in rng.integers(0, 256, size=(count, 4, 4, 3), dtype=np.uint8)])


def test_lossless_matching_recovers_originals_from_jpeg():
    rng = np.random.default_rng(0)
    originals, labels = smooth_images(rng, 60), np.arange(60) % 5
    picked = rng.permutation(60)[:20]
    queries = []
    for i in picked:
        buffer = io.BytesIO()
        Image.fromarray(originals[i]).save(buffer, format="JPEG", quality=90)
        queries.append(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")))
    index, best, second = vtab_lossless.match(np.stack(queries), labels[picked], originals, labels)
    assert (index == picked).all() and vtab_lossless.trusted(index, best, second).all()
    twins = np.concatenate([originals[:1], originals[:1]])
    index, best, second = vtab_lossless.match(originals[:1], labels[:1], twins, np.array([labels[0]] * 2))
    assert not vtab_lossless.trusted(index, best, second).any()
    reused = vtab_lossless.trusted(np.array([3, 3, 4]), np.array([1.0, 1.0, 1.0]), np.array([30.0, 30.0, 30.0]))
    assert reused.tolist() == [False, False, True]


def test_lossless_copy_rewrites_lists_with_original_pixels(tmp_path):
    rng = np.random.default_rng(3)
    originals = {"train": (smooth_images(rng, 40), np.arange(40) % 4), "test": (smooth_images(rng, 20), np.arange(20) % 4)}
    root, manifests = tmp_path / "vtab-1k", tmp_path / "vtab_manifests"
    picks = {"train800": ("train", [3, 7, 11, 20]), "val200": ("train", [30, 33]), "test": ("test", [1, 2, 5])}
    rows = {}
    for name, (source, indices) in picks.items():
        for j, i in enumerate(indices):
            path = f"svhn/images/{name}/{j:06d}.jpg"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(originals[source][0][i]).save(root / path, quality=90)
            rows.setdefault(name, []).append((path, int(originals[source][1][i])))
    manifests.mkdir()
    for protocol, splits in (("800", (("train", "train800"), ("val", "val200"), ("test", "test"))),
                             ("1000", (("train", "train800"), ("train", "val200"), ("val", "val200"), ("test", "test")))):
        with open(manifests / f"svhn_{protocol}.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "split"])
            for split, name in splits:
                writer.writerows((path, label, split) for path, label in rows[name])
    summary = vtab_lossless.build("svhn", originals, tmp_path)
    assert all(entry["trusted"] == entry["images"] for entry in summary.values())
    copied = vtab_lossless.read_rows(manifests / "svhn_png_1000.csv")
    original = vtab_lossless.read_rows(manifests / "svhn_1000.csv")
    assert [(r["label"], r["split"]) for r in copied] == [(r["label"], r["split"]) for r in original]
    assert copied[0]["path"] == "svhn_png/images/train800/000000.png"
    np.testing.assert_array_equal(np.asarray(Image.open(root / copied[0]["path"])), originals["train"][0][3])


def test_smallnorb_reader_and_split_sources(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, size=(5, 2, 4, 4), dtype=np.uint8)
    with gzip.open(tmp_path / "m.mat.gz", "wb") as handle:
        handle.write(np.array([0x1E3D4C55, 4, 5, 2, 4, 4], dtype="<i4").tobytes() + images.tobytes())
    np.testing.assert_array_equal(vtab_lossless.read_norb(tmp_path / "m.mat.gz"), images)
    labels = np.arange(4, dtype="<i4")
    with gzip.open(tmp_path / "c.mat.gz", "wb") as handle:
        handle.write(np.array([0x1E3D4C54, 1, 4, 1, 1], dtype="<i4").tobytes() + labels.tobytes())
    np.testing.assert_array_equal(vtab_lossless.read_norb(tmp_path / "c.mat.gz"), labels)
    loader, mode, sources, min_gap = vtab_lossless.TASKS["smallnorb_azi"]
    assert mode == "L" and sources["val200"] == ("test",) and sources["train800val200"] == ("train", "test")
    assert min_gap < vtab_lossless.TASKS["cifar"][3]
    assert vtab_lossless.trusted(np.array([0]), np.array([1.0]), np.array([2.5]), min_gap=min_gap).all()
    assert not vtab_lossless.trusted(np.array([0]), np.array([1.0]), np.array([1.5]), min_gap=min_gap).any()
    assert not vtab_lossless.trusted(np.array([0]), np.array([0.0]), np.array([0.0]), min_gap=min_gap).any()


def test_smallnorb_elevation_reuses_the_azimuth_copy(tmp_path):
    root, manifests = tmp_path / "vtab-1k", tmp_path / "vtab_manifests"
    manifests.mkdir()
    for task, label in (("smallnorb_azi", 7), ("smallnorb_ele", 3)):
        path = root / task / "images" / "test" / "000000.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same jpeg bytes")
        for protocol in ("800", "1000"):
            with open(manifests / f"{task}_{protocol}.csv", "w", newline="") as stream:
                csv.writer(stream).writerows([["path", "label", "split"],
                                              [f"{task}/images/test/000000.jpg", label, "test"]])
    png = root / "smallnorb_azi_png" / "images" / "test" / "000000.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    vtab_lossless.build_alias("smallnorb_ele", "smallnorb_azi", tmp_path)
    rows = vtab_lossless.read_rows(manifests / "smallnorb_ele_png_800.csv")
    assert rows == [{"path": "smallnorb_azi_png/images/test/000000.png", "label": "3", "split": "test"}]
    (root / "smallnorb_ele" / "images" / "test" / "000000.jpg").write_bytes(b"different")
    with pytest.raises(SystemExit):
        vtab_lossless.build_alias("smallnorb_ele", "smallnorb_azi", tmp_path)
