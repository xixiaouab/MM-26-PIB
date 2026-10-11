"""Lossless copy of VTAB-1k CIFAR-100 (the first task done; scripts/vtab_lossless.py also handles SVHN).

See vtab_lossless.py for the matching rules; this entry point builds cifar_png exactly as before.
"""

import argparse
from pathlib import Path

from vtab_lossless import P, build, load_cifar, match, trusted

__all__ = ["build", "load_cifar", "main", "match", "trusted"]  # match and trusted are re-exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=P + "/data")
    args = parser.parse_args()
    data = Path(args.data)
    build("cifar", load_cifar(data), data)


if __name__ == "__main__":
    main()
