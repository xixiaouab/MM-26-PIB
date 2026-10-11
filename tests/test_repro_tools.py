import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from pib.data import ImageDataset, ImageTransform, read_manifest
from pib.model import PromptedVisionTransformer, load_model_state, model_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect  # noqa: E402
import make_queue  # noqa: E402
import run_task  # noqa: E402

TINY = {"img_size": 32, "patch_size": 8, "embed_dim": 24, "depth": 2, "num_heads": 3}


def tiny_model(seed):
    import timm

    torch.manual_seed(seed)
    return PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                     prompt_length=2)


def test_vtab_transform_is_a_plain_resize():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8))
    transform = ImageTransform(True, image_size=32, resize_size=32, augment="none", resize_mode="square")
    assert transform.deterministic and transform.horizontal_flip == 0
    first, second = transform(image), transform(image)
    assert first.size == (32, 32) and np.array_equal(np.asarray(first), np.asarray(second))
    crop = ImageTransform(True, image_size=32, resize_size=36, augment="crop")
    assert crop(image).size == (32, 32) and not crop.deterministic
    with pytest.raises(ValueError):
        ImageTransform(True, augment="mixup")


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


def test_run_task_trains_and_evaluates_the_requested_checkpoint(tmp_path):
    manifest = write_images(tmp_path)
    config = {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "prompt_length": 2,
                        "backbone_kwargs": TINY},
              "data": {"root": str(tmp_path), "manifest": str(manifest), "image_size": 32, "resize_size": 32,
                       "augment": "none", "resize_mode": "square", "num_workers": 0, "cache_in_memory": True,
                       "verify_files": False},
              "training": {"epochs": 2, "warmup_epochs": 1, "batch_size": 6, "seed": 0, "learning_rate": 1e-3,
                           "regularization_warmup": 1},
              "loss": {"routing": True}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    task = {"name": "t", "config": str(config_path), "set": {"training.weight_decay": 0.01}, "eval": "best"}
    result = run_task.run(task, tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["test_samples"] == 3 and 0 <= saved["test_top1"] <= 100
    assert result["best_epoch"] in (1, 2) and saved["task"]["set"]["training.weight_decay"] == 0.01


def test_selection_uses_validation_accuracy_only():
    base = {"method": "pib", "task": "dtd", "wd": "0.0001", "spec": {"set": {}}}
    rows = [
        {**base, "length": "10", "lr": "0.001", "best_val_top1": 70.0, "test_top1": 99.0},
        {**base, "length": "50", "lr": "0.001", "best_val_top1": 72.0, "test_top1": 1.0},
        {**base, "length": "10", "lr": "0.002", "tag": "ls0.3", "best_val_top1": 72.0,
         "spec": {"set": {"model.prompt_length": 10, "loss.path_weight": 0.03}}},
    ]
    chosen = collect.select(rows)["pib"]["dtd"]
    assert (chosen["length"], chosen["lr"], chosen["tag"], chosen["val"]) == (10, 0.002, "ls0.3", 72.0)
    assert chosen["set"] == {"loss.path_weight": 0.03}


def test_collector_keeps_dataset_names(tmp_path):
    run = tmp_path / "p" / "pib" / "dtd" / "L10_lr0.001_wd0.0001_pstd_s0"
    run.mkdir(parents=True)
    run.joinpath("result.json").write_text(json.dumps({"task": {"name": "x", "set": {"a": 1}}, "best_val_top1": 70.0}))
    rows = collect.load(tmp_path, ["p"])
    assert rows[0]["task"] == "dtd" and rows[0]["spec"]["set"] == {"a": 1}


def test_run_names_round_trip_through_the_queue_and_collector():
    run = make_queue.task_spec("p", "pib", "dtd", "std", 10, 0.003, 1e-4, 2, "m.csv", "c.yaml", "best",
                               {"loss.path_weight": 0.03}, "ls0.3")
    match = collect.NAME.match(run["name"].split("/", 1)[1])
    assert match and match.group("tag") == "ls0.3" and match.group("seed") == "2" and match.group("protocol") == "std"
    vpt = make_queue.task_spec("p", "vpt", "dtd", "800", 1, 0.01, 1e-4, 0, "m.csv", "c.yaml", None)
    assert vpt["set"]["loss.routing"] is False and collect.NAME.match(vpt["name"].split("/", 1)[1]).group("tag") is None
    scaled = make_queue.scaled_loss("configs/vtab_vit_b16.yaml", 0.5)
    assert scaled["loss.compression"] == {"start": 0.005, "end": 0.05} and scaled["loss.path_weight"] == 0.05


def reference_scatter(features, labels):
    """The original per-class loop, kept as the reference for the vectorized version."""
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
    from pib.losses import layer_sufficiency, scatter_statistics

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
    from pib.losses import PIBObjective
    import torch.nn.functional as F

    objective = PIBObjective(2, compression=0.0, sufficiency=0.0, path_weight=0.0, routing=False)
    assert not objective.active and PIBObjective(2).active
    logits, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
    values = objective({"logits": logits, "states": []}, labels)
    torch.testing.assert_close(values["loss"], F.cross_entropy(logits, labels))
    torch.testing.assert_close(objective(logits, labels)["loss"], values["loss"])


def test_preempted_training_resumes_to_the_same_result(tmp_path):
    import time as clock
    from pib.train import TrainingPreempted, run_training

    manifest = write_images(tmp_path)
    config = {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "prompt_length": 2,
                        "backbone_kwargs": TINY},
              "data": {"root": str(tmp_path), "manifest": str(manifest), "image_size": 32, "resize_size": 32,
                       "augment": "none", "resize_mode": "square", "num_workers": 0, "verify_files": False},
              "training": {"epochs": 3, "warmup_epochs": 1, "batch_size": 6, "seed": 0, "learning_rate": 1e-3,
                           "regularization_warmup": 1}}
    run_training(config, tmp_path / "straight", device="cpu")
    with pytest.raises(TrainingPreempted):
        run_training(config, tmp_path / "split", device="cpu", deadline=clock.time())
    assert torch.load(tmp_path / "split" / "last.pt", weights_only=True)["epoch"] == 0
    run_training(config, tmp_path / "split", resume=tmp_path / "split" / "last.pt", device="cpu")
    straight = torch.load(tmp_path / "straight" / "last.pt", weights_only=True)
    split = torch.load(tmp_path / "split" / "last.pt", weights_only=True)
    assert split["epoch"] == straight["epoch"] == 2
    for key, value in straight["model"].items():
        torch.testing.assert_close(split["model"][key], value, rtol=0, atol=0)


def test_eval_selected_matches_exact_configurations(tmp_path):
    import eval_selected

    for name in ("L10_lr0.003_wd0.0001_p800_s0", "L10_lr0.003_wd0.0001_p800_s1", "L10_lr0.003_wd0.0001_p800_s0_ls0.3",
                 "L50_lr0.003_wd0.0001_p800_s0"):
        run = tmp_path / "p" / "pib" / "dtd" / name
        run.mkdir(parents=True)
        run.joinpath("result.json").write_text(json.dumps({"task": {"set": {}}, "best_val_top1": 1.0}))
    rows = collect.load(tmp_path, ["p"], "800")
    plain = {"pib": {"dtd": {"length": 10, "lr": 0.003, "wd": 0.0001, "tag": ""}}}
    tagged = {"pib": {"dtd": {"length": 10, "lr": 0.003, "wd": 0.0001, "tag": "ls0.3"}}}
    assert sorted(Path(r["dir"]).name for r in eval_selected.chosen_runs(rows, plain)) == [
        "L10_lr0.003_wd0.0001_p800_s0", "L10_lr0.003_wd0.0001_p800_s1"]
    assert [Path(r["dir"]).name for r in eval_selected.chosen_runs(rows, tagged)] == ["L10_lr0.003_wd0.0001_p800_s0_ls0.3"]


def test_min_seeds_keeps_lucky_single_runs_out_of_the_final_choice():
    base = {"method": "vpt", "task": "kitti", "wd": "0.0001", "spec": {"set": {}}}
    rows = [{**base, "length": "1", "lr": "0.01", "seed": "0", "best_val_top1": 90.0}]
    rows += [{**base, "length": "5", "lr": "0.003", "seed": str(s), "best_val_top1": v} for s, v in enumerate((85, 84, 86))]
    assert collect.select(rows)["vpt"]["kitti"]["length"] == 1
    chosen = collect.select(rows, min_seeds=3)["vpt"]["kitti"]
    assert (chosen["length"], chosen["seeds"], chosen["val"]) == (5, 3, 85.0)
    top = collect.select(rows, top=2)["vpt"]["kitti"]
    assert [c["length"] for c in top] == [1, 5]


def test_report_counts_only_the_chosen_configuration(capsys):
    base = {"method": "pib", "task": "cub", "wd": "0.0001", "lr": "0.003", "spec": {"set": {}}}
    rows = [{**base, "length": "10", "test_top1": 90.0}, {**base, "length": "10", "test_top1": 92.0},
            {**base, "length": "50", "test_top1": 10.0}]
    collect.report(rows, "fgvc", {"pib": {"cub": {"length": 10, "lr": 0.003, "wd": 0.0001, "tag": ""}}})
    assert "| cub | 91.00 ± 1.41 | 2 |" in capsys.readouterr().out


def test_queue_orders_the_longest_runs_first(tmp_path):
    out = tmp_path / "q.jsonl"
    sys.argv = ["make_queue.py", "--benchmark", "fgvc", "--stage", "search", "--prefix", "f", "--output", str(out),
                "--tasks", "flowers", "nabirds", "--lengths", "1", "50", "--lrs", "0.01", "--longest-first"]
    make_queue.main()
    names = [json.loads(line)["name"].split("/")[2:4] for line in out.read_text().splitlines()]
    assert names[0] == ["nabirds", "L50_lr0.01_wd0.0001_pstd_s0"] and names[-1][0] == "flowers"


def test_parallel_untar_completes_a_partial_extraction(tmp_path):
    import tarfile
    import parallel_untar

    source = tmp_path / "src" / "food"
    for index in range(30):
        (source / f"c{index % 3}").mkdir(parents=True, exist_ok=True)
        (source / f"c{index % 3}" / f"{index}.jpg").write_bytes(bytes([index]) * (index + 1))
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="food")
    destination = tmp_path / "out"
    (destination / "food" / "c0").mkdir(parents=True)
    (destination / "food" / "c0" / "0.jpg").write_bytes(b"\x00")      # already complete
    (destination / "food" / "c0" / "3.jpg").write_bytes(b"\x03")      # truncated, must be rewritten
    written = sum(parallel_untar.extract_share(archive, destination, w, 4) for w in range(4))
    assert written == 29
    for index in range(30):
        assert (destination / "food" / f"c{index % 3}" / f"{index}.jpg").read_bytes() == bytes([index]) * (index + 1)


@pytest.mark.parametrize("options", [{"head": "mlp", "head_hidden": 16, "head_dropout": 0.1}, {"features": "cls_gap"},
                                     {"prompt_reparam": {"dim": 8, "hidden": 16}},
                                     {"head": "mlp", "features": "cls_gap", "prompt_reparam": {"dim": 8, "hidden": 16}}])
def test_head_feature_and_reparam_variants_train_only_new_parameters(options):
    import timm

    torch.manual_seed(0)
    model = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, **options)
    images = torch.randn(4, 3, 32, 32)
    output = model(images, return_states=True)
    assert output["logits"].shape == (4, 3) and len(output["states"]) == 2
    output["logits"].sum().backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert not any(name.startswith("backbone.") for name in trained)
    if "prompt_reparam" in options:
        assert {"prompt_mlp.0.weight", "prompt_embeddings.1"} <= trained and model.prompts is None
    state = model_state(model, include_backbone=False)
    clone = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, **options)
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, state, backbone_in_checkpoint=False)
    model.eval(), clone.eval()
    torch.testing.assert_close(clone(images), model(images))


def test_swin_rejects_vit_only_variants():
    import timm

    swin = timm.create_model("swin_tiny_patch4_window7_224", num_classes=0, img_size=32, patch_size=2, window_size=4,
                             embed_dim=12, depths=(1, 1), num_heads=(1, 2))
    with pytest.raises(ValueError):
        PromptedVisionTransformer(swin, 3, prompt_length=2, features="cls_gap")


def test_variant_tags_extend_the_selected_configuration(tmp_path):
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"pib": {"dtd": {"length": 10, "lr": 0.01, "wd": 0.0001, "tag": "ls0.3",
                                                     "set": {"loss.path_weight": 0.03}}}}))
    out = tmp_path / "q.jsonl"
    sys.argv = ["make_queue.py", "--benchmark", "vtab", "--stage", "refine", "--prefix", "v", "--methods", "pib",
                "--tasks", "dtd", "--selection", str(selection), "--seeds", "0", "--output", str(out),
                "--set", 'model.head="mlp"', "--tag", "mlp"]
    make_queue.main()
    run = json.loads(out.read_text())
    assert run["name"].endswith("_s0_ls0.3-mlp")
    assert run["set"]["model.head"] == "mlp" and run["set"]["loss.path_weight"] == 0.03
    assert collect.NAME.match(run["name"].split("/", 1)[1]).group("tag") == "ls0.3-mlp"


def test_compare_variants_scores_base_and_variants_on_validation(tmp_path, capsys):
    import compare_variants

    def write(prefix, name, val):
        run = tmp_path / prefix / "pib" / "flowers" / name
        run.mkdir(parents=True)
        run.joinpath("result.json").write_text(json.dumps({"task": {"set": {}}, "best_val_top1": val, "test_top1": 0.0}))
    for seed, val in enumerate((90.0, 92.0)):
        write("base", f"L1_lr0.003_wd0.0001_pstd_s{seed}_ls0.3", val)
        write("var", f"L1_lr0.003_wd0.0001_pstd_s{seed}_ls0.3-mlp", val + 2)
    write("var", "L10_lr0.003_wd0.0001_pstd_s0_ls0.3-mlp", 99.0)  # a different configuration: ignored
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"pib": {"flowers": {"length": 1, "lr": 0.003, "wd": 0.0001, "tag": "ls0.3"}}}))
    sys.argv = ["compare_variants.py", "--benchmark", "fgvc", "--selection", str(selection), "--runs", str(tmp_path),
                "--base-prefix", "base", "--prefix", "var", "--protocol", "std"]
    compare_variants.main()
    assert "| flowers | 91.0 | 93.0 |" in capsys.readouterr().out


def test_vpt_recipe_options_sgd_and_xavier_prompts(tmp_path):
    import math
    import timm

    model = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=4, prompt_init="xavier")
    bound = math.sqrt(6.0 / (3 * 8 * 8 + 24))
    assert all(float(p.abs().max()) <= bound for p in model.prompts.values())
    manifest = write_images(tmp_path)
    config = {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "prompt_length": 2,
                        "backbone_kwargs": TINY, "prompt_init": "xavier"},
              "data": {"root": str(tmp_path), "manifest": str(manifest), "image_size": 32, "resize_size": 32,
                       "augment": "none", "resize_mode": "square", "num_workers": 0, "verify_files": False},
              "training": {"epochs": 2, "warmup_epochs": 1, "batch_size": 6, "seed": 0, "learning_rate": 0.25,
                           "optimizer": "sgd", "weight_decay": 0.001}}
    from pib.train import run_training
    result = run_training(config, tmp_path / "sgd", device="cpu")
    assert result["epochs"] == 2
    state = torch.load(tmp_path / "sgd" / "last.pt", weights_only=True)
    assert "momentum_buffer" in str(state["optimizer"]["state"])


def test_flip_augmentation_keeps_the_evaluation_geometry():
    image = Image.fromarray(np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3))
    flip = ImageTransform(True, image_size=32, resize_size=32, augment="flip", resize_mode="square", horizontal_flip=1.0)
    plain = ImageTransform(False, image_size=32, resize_size=32, augment="flip", resize_mode="square")
    assert not flip.deterministic and plain.deterministic
    assert np.array_equal(np.asarray(flip(image)), np.asarray(plain(image))[:, ::-1])


def test_loss_scale_refinement_keeps_other_tags(tmp_path):
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"pib": {"dtd": {"length": 1, "lr": 0.625, "wd": 0.001, "tag": "sgd-ls0.3",
                                                     "set": {"training.optimizer": "sgd", "loss.path_weight": 0.03}}}}))
    out = tmp_path / "q.jsonl"
    sys.argv = ["make_queue.py", "--benchmark", "vtab", "--stage", "refine", "--prefix", "b", "--tasks", "dtd",
                "--selection", str(selection), "--seeds", "0", "--loss-scale", "0.1", "--output", str(out)]
    make_queue.main()
    run = json.loads(out.read_text())
    assert run["name"].endswith("_s0_sgd-ls0.1")
    assert run["set"]["training.optimizer"] == "sgd" and run["set"]["loss.path_weight"] == 0.01


def test_drivers_run_after_earlier_tasks_and_extend_the_queue(tmp_path):
    import subprocess as sp

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    queue, runs, log = tmp_path / "queue.jsonl", tmp_path / "runs", tmp_path / "order.txt"
    step = lambda name, sleep=0.0: {"name": name, "cmd": f"sleep {sleep}; echo {name} >> {log}"}
    stage2 = [step("b1", 0.3), step("b2", 0.1)]
    appender = (f"{sys.executable} -c \"import sys; sys.path.insert(0, '{scripts}'); from queue_append import append_tasks; "
                f"append_tasks('{queue}', {stage2!r})\" && echo driver >> {log}")
    queue.write_text("".join(json.dumps(t) + "\n" for t in [step("a1", 0.6), step("a2", 0.2),
                                                             {"name": "d", "driver": True, "cmd": appender}]))
    workers = [sp.Popen([sys.executable, str(scripts / "queue_worker.py"), "--queue", str(queue), "--runs", str(runs),
                         "--poll-seconds", "0.1"], cwd=tmp_path, stdout=sp.DEVNULL) for _ in range(3)]
    assert all(w.wait(timeout=60) == 0 for w in workers)
    order = log.read_text().split()
    assert sorted(order) == ["a1", "a2", "b1", "b2", "driver"]
    assert order.index("driver") > max(order.index("a1"), order.index("a2"))
    assert all((runs / name / "result.json").exists() for name in ("a1", "a2", "b1", "b2", "d"))


def test_lora_and_adapter_start_as_the_plain_model_and_train_outside_the_backbone():
    import timm

    torch.manual_seed(0)
    options = {"lora": {"rank": 2, "targets": ["qkv", "fc1"]}, "adapter": {"dim": 4, "scale": 0.5}}
    model = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, **options)
    plain = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3, prompt_length=2)
    plain.backbone.load_state_dict(model.backbone.state_dict())
    plain.prompts.load_state_dict(model.prompts.state_dict())
    plain.head.load_state_dict(model.head.state_dict())
    images = torch.randn(4, 3, 32, 32)
    model.eval(), plain.eval()
    torch.testing.assert_close(model(images), plain(images))  # zero-initialized updates leave the model unchanged
    model.train()
    output = model(images, return_states=True)
    assert len(output["states"]) == 2
    output["logits"].sum().backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert not any(name.startswith("backbone.") for name in trained)
    assert {"lora.1_qkv.up.weight", "lora.2_fc1.up.weight", "adapters.1.up.weight"} <= trained
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("up.weight"):
                parameter.normal_()
    model.eval()
    assert not torch.allclose(model(images), plain(images))  # the hooks are live
    state = model_state(model, include_backbone=False)
    assert any(key.startswith("lora.") for key in state) and any(key.startswith("adapters.") for key in state)
    clone = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, **options)
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, state, backbone_in_checkpoint=False)
    clone.eval()
    torch.testing.assert_close(clone(images), model(images))
    with pytest.raises(ValueError):
        PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                  prompt_length=2, lora={"rank": 2, "targets": ["unknown"]})


def test_swin_rejects_lora():
    import timm

    swin = timm.create_model("swin_tiny_patch4_window7_224", num_classes=0, img_size=32, patch_size=2, window_size=4,
                             embed_dim=12, depths=(1, 1), num_heads=(1, 2))
    with pytest.raises(ValueError):
        PromptedVisionTransformer(swin, 3, prompt_length=2, lora={"rank": 2})


def test_selection_by_last_epoch_validation_and_method_filter(tmp_path):
    rows = []
    for length, best, last in ((1, 90.0, 60.0), (10, 85.0, 84.0)):
        for seed in range(3):
            rows.append({"method": "pib", "task": "dtd", "length": str(length), "lr": "0.001", "wd": "0.0001",
                         "tag": None, "best_val_top1": best, "last_val_top1": last, "spec": {"set": {}}})
    assert collect.select(rows)["pib"]["dtd"]["length"] == 1
    assert collect.select(rows, metric="last_val_top1")["pib"]["dtd"]["length"] == 10
    for method in ("pib", "vpt"):
        run_dir = tmp_path / "p" / method / "dtd" / "L1_lr0.001_wd0.0001_p800_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"best_val_top1": 50.0, "last_val_top1": 40.0}))
    output = tmp_path / "sel.json"
    sys.argv = ["collect.py", "--benchmark", "vtab", "--runs", str(tmp_path), "--prefix", "p", "--methods", "pib",
                "--select", str(output)]
    collect.main()
    assert set(json.loads(output.read_text())) == {"pib"}


def test_lora_adapter_campaign_stages_extend_the_queue(tmp_path):
    import os
    import random
    import subprocess

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    (tmp_path / "envs" / "pib" / "bin").mkdir(parents=True)
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"  # a symlink would lose the virtual environment
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    environment = {**os.environ, "PIB_ROOT": str(tmp_path)}
    queue = tmp_path / "q.jsonl"

    def stage(name):
        subprocess.run([sys.executable, "vtab_ad.py", name, "--queue", str(queue)], cwd=scripts, env=environment,
                       check=True, capture_output=True, text=True)
        return [json.loads(line) for line in queue.read_text().splitlines()]

    def finish(tasks, rng):
        for task in tasks:
            if task.get("driver") or "cmd" in task:
                continue
            run_dir = tmp_path / "runs" / task["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps({"name": task["name"], "task": task,
                                                             "best_val_top1": rng.uniform(50, 90),
                                                             "last_val_top1": rng.uniform(40, 90)}))

    rng = random.Random(0)
    tasks = stage("search")
    search = [t for t in tasks if not t.get("driver")]
    assert len(search) == 19 * 4 * 2 * 4 * 2 and tasks[-1]["name"] == "driver_vtab_ad_refine"
    assert all(t["name"].startswith("vtab_ad1/pib/") for t in search)
    lora = next(t for t in search if t["name"].endswith("_lora8all"))
    assert lora["set"]["model.lora"] == {"rank": 8, "targets": ["qkv", "proj", "fc1", "fc2"]}
    finish(search, rng)
    tasks = stage("refine")
    refine = [t for t in tasks[len(search) + 1:] if not t.get("driver")]
    assert tasks[-1]["name"] == "driver_vtab_ad_final" and len({t["name"] for t in refine}) == len(refine)
    assert all(t["name"].startswith("vtab_ad2/pib/") for t in refine)
    assert any(t["name"].endswith("-gap") for t in refine) and any("-ls0.3" in t["name"] for t in refine)
    finish(refine, rng)
    tasks = stage("final")
    appended = tasks[len(search) + len(refine) + 2:]
    finals = [t for t in appended if t["name"].startswith("vtab_p1000ad/")]
    tests = [t for t in appended if t["name"].startswith("test_")]
    assert len(finals) == 19 * 5 and all(t["eval"] == "last" and "_p1000_" in t["name"] for t in finals)
    assert len(tests) >= 19 * 3 and appended[-1]["name"] == "driver_vtab_ad_report"
    assert "vtab_report_ad.md" in appended[-1]["cmd"]


def test_lossless_cifar_matching_recovers_originals_from_jpeg():
    import io

    import vtab_lossless_cifar

    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 256, size=(60, 4, 4, 3), dtype=np.uint8)  # smooth images, like natural photos
    originals = np.stack([np.asarray(Image.fromarray(c).resize((32, 32), Image.BILINEAR)) for c in coarse])
    labels = np.arange(60) % 5
    picked = rng.permutation(60)[:20]
    queries = []
    for i in picked:
        buffer = io.BytesIO()
        Image.fromarray(originals[i]).save(buffer, format="JPEG", quality=90)
        queries.append(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")))
    index, best, second = vtab_lossless_cifar.match(np.stack(queries), labels[picked], originals, labels)
    assert (index == picked).all() and vtab_lossless_cifar.trusted(index, best, second).all()
    twins = np.concatenate([originals[:1], originals[:1]])  # two identical candidates cannot be told apart
    index, best, second = vtab_lossless_cifar.match(originals[:1], labels[:1], twins, np.array([labels[0]] * 2))
    assert not vtab_lossless_cifar.trusted(index, best, second).any()
    reused = vtab_lossless_cifar.trusted(np.array([3, 3, 4]), np.array([1.0, 1.0, 1.0]), np.array([30.0, 30.0, 30.0]))
    assert reused.tolist() == [False, False, True]


def test_second_round_searches_larger_modules_and_longer_training(tmp_path):
    import os
    import subprocess

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    queue = tmp_path / "q.jsonl"
    subprocess.run([sys.executable, "vtab_ad.py", "search", "--round", "2", "--queue", str(queue)], cwd=scripts,
                   env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
    tasks = [json.loads(line) for line in queue.read_text().splitlines()]
    runs = tasks[:-1]
    assert len(runs) == 13 * 3 * 3 * 2 * 2 and all(t["name"].startswith("vtab_ad3/pib/") for t in runs)
    assert not any("/cifar/" in t["name"] or "/dtd/" in t["name"] for t in runs)
    long = [t for t in runs if t["name"].endswith("-ep300")]
    assert len(long) == len(runs) // 2 and all(t["set"]["training.epochs"] == 300 for t in long)
    assert runs[0]["name"].endswith("-ep300")  # longest first
    assert tasks[-1]["name"] == "driver_vtab_ad2_refine" and "--round 2" in tasks[-1]["cmd"]


def test_lossless_cifar_round_selects_only_among_its_own_runs(tmp_path):
    import os
    import random
    import subprocess

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    queue, rng = tmp_path / "q.jsonl", random.Random(1)

    def stage(name):
        subprocess.run([sys.executable, "vtab_ad.py", name, "--round", "c", "--queue", str(queue)], cwd=scripts,
                       env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
        return [json.loads(line) for line in queue.read_text().splitlines()]

    def finish(tasks):
        for task in tasks:
            if not task.get("driver") and "cmd" not in task:
                run_dir = tmp_path / "runs" / task["name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "result.json").write_text(json.dumps({"task": task, "best_val_top1": rng.uniform(60, 80),
                                                                 "last_val_top1": rng.uniform(50, 80)}))

    old = tmp_path / "runs" / "vtab_s1" / "pib" / "cifar" / "L1_lr0.001_wd0.0001_p800_s0"  # JPEG-data run: ignored
    old.mkdir(parents=True)
    (old / "result.json").write_text(json.dumps({"best_val_top1": 99.0, "last_val_top1": 99.0}))
    search = [t for t in stage("search") if not t.get("driver")]
    assert len(search) == 40 + 16 + 12 + 12 and all(t["name"].startswith("vtab_c1/pib/cifar_png/") for t in search)
    assert all(t["set"]["data.manifest"].endswith("/cifar_png_800.csv") for t in search)
    assert sum(t["set"].get("training.optimizer") == "sgd" for t in search) == 40
    finish(search)
    finish(stage("refine"))
    tasks = stage("final")
    finals = [t for t in tasks if t["name"].startswith("vtab_p1000c/")]
    assert len(finals) == 5 and all(t["set"]["data.manifest"].endswith("/cifar_png_1000.csv") for t in finals)
    assert not any("/cifar/" in t["name"] for t in tasks) and tasks[-1]["name"] == "driver_vtab_adc_report"


def test_lossless_copy_rewrites_lists_with_original_pixels(tmp_path):
    import csv as csv_module

    import vtab_lossless

    rng = np.random.default_rng(3)
    smooth = lambda n: np.stack([np.asarray(Image.fromarray(c).resize((32, 32), Image.BILINEAR))
                                 for c in rng.integers(0, 256, size=(n, 4, 4, 3), dtype=np.uint8)])
    originals = {"train": (smooth(40), np.arange(40) % 4), "test": (smooth(20), np.arange(20) % 4)}
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
            writer = csv_module.writer(stream)
            writer.writerow(["path", "label", "split"])
            for split, name in splits:
                writer.writerows((path, label, split) for path, label in rows[name])
    summary = vtab_lossless.build("svhn", originals, tmp_path)
    assert all(entry["trusted"] == entry["images"] for entry in summary.values())
    copied = vtab_lossless.read_rows(manifests / "svhn_png_1000.csv")
    original = vtab_lossless.read_rows(manifests / "svhn_1000.csv")
    assert [(r["label"], r["split"]) for r in copied] == [(r["label"], r["split"]) for r in original]
    first = copied[0]
    assert first["path"] == "svhn_png/images/train800/000000.png"
    np.testing.assert_array_equal(np.asarray(Image.open(root / first["path"])), originals["train"][0][3])


def test_smallnorb_reader_and_split_sources(tmp_path):
    import gzip

    import vtab_lossless

    images = np.random.default_rng(0).integers(0, 256, size=(5, 2, 4, 4), dtype=np.uint8)
    header = np.array([0x1E3D4C55, 4, 5, 2, 4, 4], dtype="<i4").tobytes()
    with gzip.open(tmp_path / "m.mat.gz", "wb") as handle:
        handle.write(header + images.tobytes())
    np.testing.assert_array_equal(vtab_lossless.read_norb(tmp_path / "m.mat.gz"), images)
    labels = np.arange(4, dtype="<i4")
    with gzip.open(tmp_path / "c.mat.gz", "wb") as handle:  # one-dimensional matrices still store three sizes
        handle.write(np.array([0x1E3D4C54, 1, 4, 1, 1], dtype="<i4").tobytes() + labels.tobytes())
    np.testing.assert_array_equal(vtab_lossless.read_norb(tmp_path / "c.mat.gz"), labels)
    loader, mode, sources, min_gap = vtab_lossless.TASKS["smallnorb_azi"]
    assert mode == "L" and sources["val200"] == ("test",) and sources["train800val200"] == ("train", "test")
    assert min_gap < vtab_lossless.TASKS["cifar"][3]
    # A close runner-up (same pose, other lighting) is still told apart when it is twice as far.
    assert vtab_lossless.trusted(np.array([0]), np.array([1.0]), np.array([2.5]), min_gap=min_gap).all()
    assert not vtab_lossless.trusted(np.array([0]), np.array([1.0]), np.array([1.5]), min_gap=min_gap).any()
    assert not vtab_lossless.trusted(np.array([0]), np.array([0.0]), np.array([0.0]), min_gap=min_gap).any()


def test_queue_reader_skips_malformed_lines(tmp_path, capsys):
    import queue_worker

    queue = tmp_path / "q.jsonl"
    queue.write_text('{"name": "a"}\n{"name": "b", "cmd": "x \\& y"}\n\n{"name": "c"}\n{"name": "d"')
    assert [task["name"] for task in queue_worker.read_queue(queue)] == ["a", "c"]
    assert "line 2" in capsys.readouterr().err


def test_commands_running_into_the_deadline_release_their_claim(tmp_path):
    import subprocess
    import time

    queue, runs = tmp_path / "q.jsonl", tmp_path / "runs"
    queue.write_text(json.dumps({"name": "slow", "cmd": "sleep 30"}) + "\n")
    started = time.time()
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "queue_worker.py"),
                    "--queue", str(queue), "--runs", str(runs), "--deadline", str(time.time() + 63),
                    "--reserve-minutes", "0"], check=True, capture_output=True, text=True, timeout=60)
    assert time.time() - started < 30
    assert not any((runs / "slow" / name).exists() for name in ("claim", "result.json", "failed"))


def test_smallnorb_elevation_reuses_the_azimuth_copy(tmp_path):
    import csv as csv_module

    import vtab_lossless

    root, manifests = tmp_path / "vtab-1k", tmp_path / "vtab_manifests"
    manifests.mkdir()
    for task, label in (("smallnorb_azi", 7), ("smallnorb_ele", 3)):
        path = root / task / "images" / "test" / "000000.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same jpeg bytes")
        for protocol in ("800", "1000"):
            with open(manifests / f"{task}_{protocol}.csv", "w", newline="") as stream:
                csv_module.writer(stream).writerows([["path", "label", "split"],
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


def test_label_smoothing_applies_to_the_task_loss_only():
    from torch.nn import functional as F

    from pib.losses import PIBObjective

    logits, labels = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 0, 1])
    plain = PIBObjective(2, compression={"start": 0, "end": 0}, sufficiency={"start": 0, "end": 0}, path_weight=0,
                         label_smoothing=0.1)
    torch.testing.assert_close(plain(logits, labels)["loss"], F.cross_entropy(logits, labels, label_smoothing=0.1))
    default = PIBObjective(2, compression={"start": 0, "end": 0}, sufficiency={"start": 0, "end": 0}, path_weight=0)
    torch.testing.assert_close(default(logits, labels)["loss"], F.cross_entropy(logits, labels))
    with pytest.raises(ValueError):
        PIBObjective(2, label_smoothing=1.0)


def test_label_smoothing_round_covers_every_task_and_reuses_final_runs(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    environment, rng, queue = {**os.environ, "PIB_ROOT": str(tmp_path)}, random.Random(2), tmp_path / "q.jsonl"
    tasks = benchmarks.tasks("vtab_png")
    for task in tasks:
        for seed in range(3):
            run_dir = tmp_path / "runs" / "vtab_ad1" / "pib" / task / f"L1_lr0.001_wd0.0001_p800_s{seed}_ad32"
            run_dir.mkdir(parents=True)
            spec = {"set": {"model.adapter": {"dim": 32, "scale": 0.1}}}
            (run_dir / "result.json").write_text(json.dumps({"task": spec, "best_val_top1": rng.uniform(50, 90),
                                                             "last_val_top1": rng.uniform(40, 90)}))
    reused = tmp_path / "runs" / "vtab_p1000c" / "pib" / "cifar_png" / "L1_lr0.001_wd0.0001_p1000_s0_ad32"
    reused.mkdir(parents=True)
    (reused / "result.json").write_text(json.dumps({"best_val_top1": 1.0, "last_val_top1": 1.0, "test_top1": 80.0}))
    subprocess.run([sys.executable, "vtab_variants.py", "--queue", str(queue)], cwd=scripts, env=environment,
                   check=True, capture_output=True, text=True)
    appended = [json.loads(line) for line in queue.read_text().splitlines()]
    runs = appended[:-1]
    assert len(runs) == 19 * 3 and all(t["name"].endswith("_ad32-smooth0.1") for t in runs)
    assert all(t["set"]["loss.label_smoothing"] == 0.1 and "model.adapter" in t["set"] for t in runs)
    assert {t["name"].split("/")[2] for t in runs} == set(tasks) and appended[-1]["name"] == "driver_vtab_ade_final"


def test_cross_validation_folds_partition_the_training_images(tmp_path):
    import csv as csv_module
    from collections import Counter

    import benchmarks
    import vtab_cv_folds

    for task in benchmarks.tasks("vtab_png"):
        with open(tmp_path / f"{task}_1000.csv", "w", newline="") as stream:
            writer = csv_module.writer(stream)
            writer.writerow(["path", "label", "split"])
            writer.writerows((f"{task}/img{i}.jpg", i % 37, "train") for i in range(1000))
            writer.writerows((f"{task}/test{i}.jpg", i % 37, "test") for i in range(50))
        with open(tmp_path / f"{task}_800.csv", "w", newline="") as stream:
            writer = csv_module.writer(stream)
            writer.writerow(["path", "label", "split"])
            writer.writerows((f"{task}/train800/{i}.jpg", i % 9, "train") for i in range(800))
            writer.writerows((f"{task}/val200/{i}.jpg", i % 9, "val") for i in range(200))
            writer.writerows((f"{task}/test{i}.jpg", i % 9, "test") for i in range(50))
    sys.argv = ["vtab_cv_folds.py", "--manifests", str(tmp_path)]
    vtab_cv_folds.main()
    held_out = [vtab_lossless_rows(tmp_path / f"smallnorb_azi_png_cvv{k}.csv") for k in range(5)]
    assert all(sum(r["split"] == "val" for r in rows) == 40 and all("val200/" in r["path"] for r in rows if r["split"] == "val")
               and sum("train800/" in r["path"] and r["split"] == "train" for r in rows) == 800 for rows in held_out)
    assert len({r["path"] for rows in held_out for r in rows if r["split"] == "val"}) == 200
    assert not (tmp_path / "dtd_cvv0.csv").exists()
    folds = [vtab_lossless_rows(tmp_path / f"dtd_cv{k}.csv") for k in range(5)]
    vals = [{row["path"] for row in rows if row["split"] == "val"} for rows in folds]
    assert [len(v) for v in vals] == [200] * 5 and len(set().union(*vals)) == 1000
    assert all(len(rows) == 1000 and not any(row["split"] == "test" for row in rows) for rows in folds)
    per_label = [Counter(int(row["label"]) for row in rows if row["split"] == "val") for rows in folds]
    assert all(max(c[label] for c in per_label) - min(c[label] for c in per_label) <= 1 for label in range(37))


def vtab_lossless_rows(path):
    import csv as csv_module

    with open(path, newline="") as stream:
        return list(csv_module.DictReader(stream))


def test_cross_validation_round_selects_by_fold_means(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    environment, rng, queue = {**os.environ, "PIB_ROOT": str(tmp_path)}, random.Random(4), tmp_path / "q.jsonl"

    def write_result(run_dir, spec):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps({"task": spec, "best_val_top1": rng.uniform(50, 90),
                                                         "last_val_top1": rng.uniform(40, 90)}))

    for task in benchmarks.tasks("vtab_png"):
        for lr in ("0.001", "0.003"):
            for seed in range(3):
                write_result(tmp_path / "runs" / "vtab_ad1" / "pib" / task / f"L1_lr{lr}_wd0.0001_p800_s{seed}_ad32",
                             {"set": {"model.adapter": {"dim": 32, "scale": 0.1}}})

    def stage(name):
        subprocess.run([sys.executable, "vtab_cv.py", name, "--queue", str(queue)], cwd=scripts, env=environment,
                       check=True, capture_output=True, text=True)
        return [json.loads(line) for line in queue.read_text().splitlines()]

    tasks = stage("start")
    folds = tasks[:-1]
    assert len(folds) == 19 * 2 * 5 and tasks[-1]["name"] == "driver_vtab_cv_finish"
    assert all("_pcv" in t["name"] and t["set"]["data.manifest"].endswith(t["name"].split("_p")[-1].split("_")[0] + ".csv")
               for t in folds)
    for task in folds:
        write_result(tmp_path / "runs" / task["name"], task)
    tasks = stage("finish")
    finals = [t for t in tasks[len(folds) + 1:] if not t.get("driver")]
    assert len(finals) == 19 * 5 and all(t["name"].startswith("vtab_p1000cv/") and t["eval"] == "last" for t in finals)
    assert all(t["set"]["data.manifest"].endswith("_1000.csv") for t in finals) and tasks[-1]["name"] == "driver_vtab_cv_report"


def test_second_cross_validation_round_uses_val200_folds_for_smallnorb(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    environment, rng, queue = {**os.environ, "PIB_ROOT": str(tmp_path)}, random.Random(5), tmp_path / "q.jsonl"

    def write_result(run_dir, spec, last=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps({"task": spec, "best_val_top1": rng.uniform(50, 90),
                                                         "last_val_top1": rng.uniform(40, 90) if last is None else last}))

    spec = {"set": {"model.adapter": {"dim": 32, "scale": 0.1}}}
    for task in benchmarks.tasks("vtab_png"):
        for lr in ("0.001", "0.003", "0.01"):
            for seed in range(3):
                write_result(tmp_path / "runs" / "vtab_ad1" / "pib" / task / f"L1_lr{lr}_wd0.0001_p800_s{seed}_ad32", spec)
        for fold in range(5):  # round 1 already ran folds of the lr 0.001 configuration
            write_result(tmp_path / "runs" / "vtab_cv1" / "pib" / task / f"L1_lr0.001_wd0.0001_pcv{fold}_s0_ad32", spec,
                         last=99.0 if task.startswith("smallnorb") else None)

    def stage(name):
        subprocess.run([sys.executable, "vtab_cv.py", name, "--round", "2", "--queue", str(queue)], cwd=scripts,
                       env=environment, check=True, capture_output=True, text=True)
        return [json.loads(line) for line in queue.read_text().splitlines()]

    tasks = stage("start")
    folds = tasks[:-1]
    norb = [t for t in folds if "/smallnorb_" in t["name"]]
    assert len(norb) == 2 * 3 * 5 and all("_pcvv" in t["name"] and "_cvv" in t["set"]["data.manifest"] for t in norb)
    others = [t for t in folds if "/smallnorb_" not in t["name"]]
    assert len(others) == 17 * 2 * 5 and not any("lr0.001_" in t["name"] for t in others)  # reused from round 1
    assert tasks[-1]["name"] == "driver_vtab_cv2_finish"
    for task in folds:
        write_result(tmp_path / "runs" / task["name"], task)
    tasks = stage("finish")
    chosen = json.loads((tmp_path / "cv2_sel_cv_p1000.json").read_text())["pib"]
    assert set(chosen) == set(benchmarks.tasks("vtab_png"))
    finals = [t for t in tasks[len(folds) + 1:] if not t.get("driver")]
    assert all(t["name"].startswith("vtab_p1000cv2/") for t in finals) and tasks[-1]["name"] == "driver_vtab_cv2_report"


def test_ssf_starts_as_the_plain_model_and_trains_only_scale_and_shift():
    import timm

    torch.manual_seed(0)
    model = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, ssf=True)
    plain = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3, prompt_length=2)
    plain.backbone.load_state_dict(model.backbone.state_dict())
    plain.prompts.load_state_dict(model.prompts.state_dict())
    plain.head.load_state_dict(model.head.state_dict())
    images = torch.randn(4, 3, 32, 32)
    model.eval(), plain.eval()
    torch.testing.assert_close(model(images), plain(images))
    assert len(model.ssf) == 2 + 6 * len(model.backbone.blocks) and model.ssf["1_qkv"].scale.numel() == 3 * 24
    model.train()
    model(images, return_states=True)["logits"].sum().backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert not any(name.startswith("backbone.") for name in trained)
    assert {"ssf.patch_embed.scale", "ssf.1_fc1.shift", "ssf.norm.scale"} <= trained
    with torch.no_grad():
        model.ssf["2_fc2"].scale.mul_(1.5)
    model.eval()
    assert not torch.allclose(model(images), plain(images))
    clone = PromptedVisionTransformer(timm.create_model("vit_tiny_patch16_224", num_classes=0, **TINY), 3,
                                      prompt_length=2, ssf=True)
    clone.backbone.load_state_dict(model.backbone.state_dict())
    load_model_state(clone, model_state(model, include_backbone=False), backbone_in_checkpoint=False)
    clone.eval()
    torch.testing.assert_close(clone(images), model(images))


def test_ssf_round_searches_every_task_and_cross_validation_round_three_sees_it(tmp_path):
    import os
    import subprocess

    import vtab_cv

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    queue = tmp_path / "q.jsonl"
    subprocess.run([sys.executable, "vtab_ad.py", "search", "--round", "f", "--queue", str(queue)], cwd=scripts,
                   env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
    runs = [json.loads(line) for line in queue.read_text().splitlines()][:-1]
    assert len(runs) == 19 * 2 * 2 * 4 * 2 and all(t["set"]["model.ssf"] is True for t in runs)
    assert all(t["name"].startswith("vtab_ssf1/pib/") for t in runs)
    assert any("/cifar_png/" in t["name"] for t in runs) and not any("/cifar/" in t["name"] for t in runs)
    assert "vtab_ssf2" in vtab_cv.candidates(vtab_cv.ROUNDS["3"]["candidates"])


def test_augmented_cached_dataset_does_not_reread_files(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, size=(40, 30, 3), dtype=np.uint8)).save(path)
    records = read_manifest_rows(tmp_path, [(str(path), 0)])
    transform = ImageTransform(training=True, image_size=32, resize_size=32, augment="rrc", crop_scale=(0.5, 1.0),
                               resize_mode="square")
    dataset = ImageDataset(records, transform, num_classes=1, cache=True)
    first = dataset[0]["images"]
    path.unlink()  # a second access must come from the decoded copy in memory
    second = dataset[0]["images"]
    assert first.shape == second.shape == (3, 32, 32)


def read_manifest_rows(root, items):
    import csv as csv_module

    manifest = root / "m.csv"
    with open(manifest, "w", newline="") as stream:
        writer = csv_module.writer(stream)
        writer.writerow(["path", "label", "split"])
        writer.writerows((path, label, "train") for path, label in items)
    return read_manifest(manifest, root, verify_files=False)


def test_augmentation_round_varies_only_label_invariant_tasks(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    rng, queue = random.Random(6), tmp_path / "q.jsonl"
    for task in benchmarks.tasks("vtab_png"):
        for seed in range(3):
            run_dir = tmp_path / "runs" / "vtab_ssf1" / "pib" / task / f"L1_lr0.01_wd0.0001_p800_s{seed}_ssf"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"task": {"set": {"model.ssf": True}},
                                                             "best_val_top1": rng.uniform(50, 90),
                                                             "last_val_top1": rng.uniform(40, 90)}))
    subprocess.run([sys.executable, "vtab_variants.py", "--round", "g", "--queue", str(queue)], cwd=scripts,
                   env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
    tasks = [json.loads(line) for line in queue.read_text().splitlines()]
    runs = tasks[:-1]
    assert len(runs) == 10 * 2 * 3 and tasks[-1]["name"] == "driver_vtab_adg_final"
    assert {t["name"].split("/")[2] for t in runs} == {"cifar_png", "caltech101", "dtd", "oxford_flowers102",
                                                        "oxford_iiit_pet", "sun397", "patch_camelyon", "eurosat",
                                                        "resisc45", "diabetic_retinopathy"}
    crops = [t for t in runs if t["name"].endswith("-rrc")]
    assert len(crops) == 30 and all(t["set"]["data.crop_scale"] == [0.5, 1.0] and t["set"]["model.ssf"] for t in crops)


def test_variant_groups_apply_their_own_variants(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    rng, queue = random.Random(7), tmp_path / "q.jsonl"
    for task in benchmarks.tasks("vtab_png"):
        for seed in range(3):
            run_dir = tmp_path / "runs" / "vtab_ssf1" / "pib" / task / f"L1_lr0.01_wd0.0001_p800_s{seed}_ssf"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"task": {"set": {"model.ssf": True}},
                                                             "best_val_top1": rng.uniform(50, 90),
                                                             "last_val_top1": rng.uniform(40, 90)}))
    subprocess.run([sys.executable, "vtab_variants.py", "--round", "h", "--queue", str(queue)], cwd=scripts,
                   env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
    runs = [json.loads(line) for line in queue.read_text().splitlines()][:-1]
    flips = {t["name"].split("/")[2] for t in runs if t["name"].endswith("-flip")}
    crops = {t["name"].split("/")[2] for t in runs if t["name"].endswith("-rrc25")}
    assert flips == {"clevr_count", "clevr_dist", "dmlab", "kitti", "smallnorb_ele_png"}
    assert len(crops) == 10 and not flips & crops and len(runs) == (5 + 10) * 3


def test_dihedral_augmentation_keeps_size_and_only_permutes_pixels():
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8))
    transform = ImageTransform(training=True, image_size=32, resize_size=32, augment="d4", resize_mode="square")
    torch.manual_seed(0)
    seen = set()
    for _ in range(40):
        out = np.asarray(transform(image))
        assert out.shape == (32, 32, 3) and sorted(out.ravel().tolist()) == sorted(np.asarray(image).ravel().tolist())
        seen.add(out.tobytes())
    assert len(seen) == 8  # all eight symmetries of the square occur
    assert not ImageTransform(training=True, augment="d4", resize_mode="square").deterministic
    evaluation = ImageTransform(training=False, image_size=32, resize_size=32, augment="d4", resize_mode="square")
    np.testing.assert_array_equal(np.asarray(evaluation(image)), np.asarray(image))


def test_cross_validation_round_eight_adds_folds_only_for_structured_tasks(tmp_path):
    import os
    import random
    import subprocess

    import benchmarks

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    wrapper = tmp_path / "envs" / "pib" / "bin" / "python"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    rng, queue = random.Random(8), tmp_path / "q.jsonl"
    for task in benchmarks.tasks("vtab_png"):
        for lr in ("0.001", "0.003"):
            for seed in range(3):
                run_dir = tmp_path / "runs" / "vtab_ad1" / "pib" / task / f"L1_lr{lr}_wd0.0001_p800_s{seed}_ad32"
                run_dir.mkdir(parents=True)
                (run_dir / "result.json").write_text(json.dumps({"task": {"set": {"model.adapter": {"dim": 32}}},
                                                                 "best_val_top1": rng.uniform(50, 90),
                                                                 "last_val_top1": rng.uniform(40, 90)}))
    subprocess.run([sys.executable, "vtab_cv.py", "start", "--round", "8", "--queue", str(queue)], cwd=scripts,
                   env={**os.environ, "PIB_ROOT": str(tmp_path)}, check=True, capture_output=True, text=True)
    runs = [json.loads(line) for line in queue.read_text().splitlines()][:-1]
    assert {t["name"].split("/")[2] for t in runs} == {"clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc",
                                                        "dsprites_ori", "smallnorb_azi_png", "smallnorb_ele_png"}
    assert len(runs) == 8 * 2 * 5
