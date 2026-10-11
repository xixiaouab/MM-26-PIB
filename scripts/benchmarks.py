"""Benchmark definitions: task groups, manifests, base configs, and the paper's PIB numbers (ViT-B/16)."""

from vtab_manifests import GROUPS as VTAB_GROUPS

DATA = "/lustre/orion/csc662/proj-shared/xxiao/PIB/data"

BENCHMARKS = {
    "vtab": {
        "groups": VTAB_GROUPS,
        "manifest": DATA + "/vtab_manifests/{task}_{protocol}.csv",
        "config": "configs/vtab_vit_b16.yaml",
        "protocols": ("800", "1000"),
        # Table S3; the paper's VTAB-1k mean (77.33) is the mean of the three group means.
        "paper": {"cifar": 80.2, "caltech101": 92.0, "dtd": 75.0, "oxford_flowers102": 99.3,
                  "oxford_iiit_pet": 91.0, "svhn": 88.36, "sun397": 55.0, "patch_camelyon": 85.0,
                  "eurosat": 96.9, "resisc45": 84.9, "diabetic_retinopathy": 76.5, "clevr_count": 84.2,
                  "clevr_dist": 71.0, "dmlab": 52.6, "kitti": 81.2, "dsprites_loc": 82.3,
                  "dsprites_ori": 50.0, "smallnorb_azi": 35.0, "smallnorb_ele": 49.1},
    },
    "fgvc": {
        "groups": {"fgvc": ["cub", "nabirds", "flowers", "dogs", "cars"]},
        "manifest": DATA + "/manifests/{task}.csv",
        "config": "configs/fgvc_vit_b16.yaml",
        "protocols": ("std",),
        "paper": {"cub": 90.4, "nabirds": 87.1, "flowers": 99.6, "dogs": 92.0, "cars": 91.4},  # Table S2
    },
    "hta": {
        "groups": {"hta": ["dtd", "cub", "nabirds", "dogs", "flowers", "food101", "cifar100", "cifar10",
                           "gtsrb", "svhn"]},
        "manifest": DATA + "/manifests/{task}.csv",
        "config": "configs/fgvc_vit_b16.yaml",
        "protocols": ("std",),
        "paper": {"dtd": 79.2, "cub": 91.0, "nabirds": 87.3, "dogs": 93.0, "flowers": 99.6, "food101": 91.1,
                  "cifar100": 93.8, "cifar10": 99.2, "gtsrb": 98.4, "svhn": 97.5},  # Table S1
    },
}


# VTAB-1k with CIFAR-100, SVHN and smallNORB from their original pixels (scripts/vtab_lossless.py) instead of JPEG copies.
LOSSLESS = {"cifar": "cifar_png", "svhn": "svhn_png", "smallnorb_azi": "smallnorb_azi_png",
            "smallnorb_ele": "smallnorb_ele_png"}
BENCHMARKS["vtab_png"] = {**BENCHMARKS["vtab"],
                          "groups": {group: [LOSSLESS.get(task, task) for task in names]
                                     for group, names in VTAB_GROUPS.items()},
                          "paper": {LOSSLESS.get(task, task): value for task, value in BENCHMARKS["vtab"]["paper"].items()}}


# Five folds of the 1,000 training images (scripts/vtab_cv_folds.py), for choosing protocol-1000 configurations.
BENCHMARKS["vtab_png_cv"] = {**BENCHMARKS["vtab_png"],
                             "protocols": tuple(f"cv{fold}" for fold in range(5)) + tuple(f"cvv{fold}" for fold in range(5))}


# Training-set sizes (after the train/val split), used to order queues by expected run time.
TRAIN_SIZES = {"cub": 5394, "nabirds": 21521, "flowers": 1020, "dogs": 10800, "cars": 7330, "dtd": 1880,
               "cifar10": 40000, "cifar100": 40000, "svhn": 58605, "gtsrb": 21312, "food101": 60600}


def expected_cost(task, prompt_length):
    """Relative run time: training images times the token count per image (197 image and class tokens)."""
    return TRAIN_SIZES.get(task, 800) * (197 + prompt_length)


def tasks(benchmark):
    return [task for group in BENCHMARKS[benchmark]["groups"].values() for task in group]
