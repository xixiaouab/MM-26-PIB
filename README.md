# Prompted Information Bottlenecks (PIB)

[Paper](https://arxiv.org/abs/2607.21973) · [MIT License](LICENSE)

## Setup

```bash
git clone https://github.com/xixiaouab/MM-26-PIB.git
cd MM-26-PIB
pip install -e .
```

Put VTAB-1k in `data/vtab-1k` (one folder per task) and the original CIFAR-100 (`cifar-100-python/`), SVHN (`svhn_{train,test}_32x32.mat`), and smallNORB (`smallnorb/`) files in `data/raw`, then build the manifests:

```bash
python scripts/vtab_manifests.py --root data/vtab-1k --output data/vtab_manifests
for task in cifar svhn smallnorb_azi smallnorb_ele; do python scripts/vtab_lossless.py --task $task; done
```

## Train

```bash
pib-train --config configs/vtab/dtd.yaml --output runs/dtd/s0 --set training.seed=0
```

## Evaluate

```bash
pib-evaluate --checkpoint runs/dtd/s0/last.pt --output runs/dtd/s0/test
```

Trained checkpoints: [pib_vtab1k_protocol1000.tar.gz](https://github.com/xixiaouab/MM-26-PIB/releases/download/vtab1k-checkpoints/pib_vtab1k_protocol1000.tar.gz). Evaluate them with `--root data/vtab-1k --manifest data/vtab_manifests/<task>_1000.csv`.
