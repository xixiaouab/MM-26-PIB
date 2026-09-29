# Prompted Information Bottlenecks (PIB)

**Rethinking Layer-Wise Information Allocation for Vision Foundation Model Adaptation**

[Paper](https://arxiv.org/abs/2607.21973) · [MIT License](LICENSE)

PIB adapts frozen vision transformers with deep prompts, layer-wise compression and sufficiency losses, a cross-layer path penalty, and optional routing gates. ViT-B/16, MAE, MoCo v3, and Swin-B configurations are included.

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
| `pib/train.py` | AdamW training, warmup, validation, and checkpoints |
| `pib/evaluate.py`, `pib/predict.py` | Test metrics and image inference |
| `pib/diagnose.py` | Layer-wise information and redundancy diagnostics |
| `configs/` | Backbone and training settings |
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
