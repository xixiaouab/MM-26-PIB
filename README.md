# Prompted Information Bottlenecks (PIB)

**Rethinking Layer-Wise Information Allocation for Vision Foundation Model Adaptation**

[Paper](https://arxiv.org/abs/2607.21973) · [MIT License](LICENSE)

PIB adapts frozen vision transformers with deep prompts, layer-wise compression and sufficiency losses, a cross-layer path penalty, and optional routing gates. ViT-B/16, MAE, MoCo v3, and Swin-B configurations are included.

## VTAB-1k checkpoints

ViT-B/16 (ImageNet-21k). PIB with LoRA/adapter/SSF modules, data augmentation, and lossless CIFAR-100, SVHN and smallNORB images; per-task settings are chosen on validation data only.

| | Natural | Specialized | Structured | Mean |
| --- | --- | --- | --- | --- |
| Paper | 82.98 | 85.83 | 63.18 | 77.33 |
| Train on 1,000 images, last epoch, 5 seeds | 83.28 | 84.89 | 61.70 | **76.62** |
| Train on 800 images, best validation epoch, 3 seeds | 81.59 | 84.45 | 57.79 | 74.61 |

[pib_vtab1k_protocol1000.tar.gz](https://github.com/xixiaouab/MM-26-PIB/releases/download/vtab1k-checkpoints/pib_vtab1k_protocol1000.tar.gz) (530 MB) holds the 95 checkpoints of the 1,000-image row (19 tasks × 5 seeds), each with its config and results. Manifests come from `scripts/vtab_manifests.py` and, for the lossless tasks, `scripts/vtab_lossless.py`.

```bash
pib-evaluate --checkpoint protocol1000/dtd/<run>/last.pt --root <vtab-1k> --manifest <manifests>/dtd_1000.csv
```

<details>
<summary>Per-task test accuracy (1,000-image row)</summary>

| Task | Ours | Paper |
| --- | --- | --- |
| CIFAR-100 | 85.39 ± 0.48 | 80.2 |
| Caltech101 | 91.60 ± 0.39 | 92.0 |
| DTD | 71.38 ± 0.96 | 75.0 |
| Flowers102 | 98.99 ± 0.09 | 99.3 |
| Pets | 90.92 ± 0.25 | 91.0 |
| SVHN | 88.04 ± 1.09 | 88.36 |
| SUN397 | 56.63 ± 0.26 | 55.0 |
| Patch Camelyon | 84.58 ± 0.35 | 85.0 |
| EuroSAT | 95.59 ± 0.33 | 96.9 |
| Resisc45 | 86.84 ± 0.58 | 84.9 |
| Retinopathy | 72.55 ± 0.55 | 76.5 |
| Clevr/count | 81.25 ± 0.89 | 84.2 |
| Clevr/distance | 64.09 ± 0.48 | 71.0 |
| DMLab | 49.74 ± 0.79 | 52.6 |
| KITTI/distance | 80.96 ± 0.58 | 81.2 |
| dSprites/location | 88.56 ± 1.37 | 82.3 |
| dSprites/orientation | 52.80 ± 0.65 | 50.0 |
| smallNORB/azimuth | 31.69 ± 0.68 | 35.0 |
| smallNORB/elevation | 44.47 ± 4.46 | 49.1 |

</details>

## Install

```bash
git clone https://github.com/xixiaouab/MM-26-PIB.git
cd MM-26-PIB
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Data

Use a CSV with `path,label,split` columns, where labels are zero-based integers and splits are `train`, `val`, and `test`. Paths are relative to `data.root`.

```csv
path,label,split
train/class_a/001.jpg,0,train
val/class_a/002.jpg,0,val
test/class_a/003.jpg,0,test
```

To create a manifest from `train/<class>/`, `val/<class>/`, and `test/<class>/` directories:

```bash
pib-prepare --root data/images --output data/splits.csv --protocol official
```

`--protocol fgvc` creates a 90/10 train/validation partition when validation is absent. `--protocol vtab` uses an 800/200 partition of the supplied 1,000 training examples. Existing official validation and test splits are preserved.

## Train and evaluate

```bash
pib-train --config configs/vit_b16.yaml --output outputs/pib \
  --set data.root=data/images data.manifest=data/splits.csv

pib-evaluate --checkpoint outputs/pib/best.pt --output outputs/pib/test
pib-predict --checkpoint outputs/pib/best.pt --image example.jpg
pib-diagnose --checkpoint outputs/pib/best.pt --output outputs/pib/layers.csv
```

The first training run downloads the selected pretrained backbone. `best.pt` is selected by validation top-1 accuracy; `last.pt` includes optimizer and random states for resuming.

```bash
pib-train --config configs/vit_b16.yaml --output outputs/pib \
  --resume outputs/pib/last.pt
```

Use `configs/mae.yaml`, `configs/moco_v3.yaml`, or `configs/swin_b.yaml` for other backbones. For MoCo v3, download the [official ViT-B checkpoint](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar) and set `model.backbone_checkpoint` to its path. Change prompt layers, loss coefficients, optimizer settings, and seeds in YAML or through `--set key=value`. Class-balanced batches use at least two samples per class where available (`training.samples_per_class`). Disable routing with `--set loss.routing=false`; set compression, sufficiency, and path weights to zero for VPT.

## Files

| File | Purpose |
| --- | --- |
| `pib/model.py` | Frozen backbones and deep prompt injection |
| `pib/losses.py` | Compression, sufficiency, path, and routing objectives |
| `pib/data.py`, `pib/prepare.py` | Image transforms, loaders, and split manifests |
| `pib/train.py` | AdamW/SGD training, warmup, validation, and checkpoints |
| `pib/evaluate.py`, `pib/predict.py` | Test metrics and image inference |
| `pib/diagnose.py` | Layer-wise information and redundancy diagnostics |
| `configs/` | Backbone and training settings |
| `scripts/` | VTAB-1k pipeline: manifests, lossless copies, Slurm queue, and validation-based selection |
| `tests/` | Equation, gradient, split, and pipeline checks |

```bash
pytest -q
```

## Citation

```bibtex
@article{li2026pib,
  title={Rethinking Layer-Wise Information Allocation for Vision Foundation Model Adaptation},
  author={Li, Yuqi and Xiao, Xi and Zhang, Yunbei and Zhao, Lin and Li, Yu and Zhao, Aiden and Wang, Tianyang and Xu, Hao and Tian, Yingli},
  journal={arXiv preprint arXiv:2607.21973},
  year={2026}
}
```
