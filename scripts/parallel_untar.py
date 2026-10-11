"""Extract a large tar archive with many small files in parallel, resuming a partial extraction.

Every worker streams the whole archive but writes only the members assigned to it (by a hash of the
member name), so file creation, the slow part on a parallel file system, is spread over the workers.
Members already present with the right size are skipped. On success the marker used by
prepare_fgvc_hta.py is written, so that script reuses the extracted files.
"""

import argparse
import tarfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def extract_share(archive, destination, worker, workers):
    written = 0
    with tarfile.open(archive) as handle:
        for member in handle:
            if zlib.crc32(member.name.encode()) % workers != worker:
                continue
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == member.size:
                continue
            handle.extract(member, destination, filter="data")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    archive, destination = Path(args.archive), Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        counts = list(pool.map(extract_share, [archive] * args.workers, [destination] * args.workers,
                               range(args.workers), [args.workers] * args.workers))
    (destination / f".extracted_{archive.name}").touch()
    print(f"wrote {sum(counts)} files with {args.workers} workers")


if __name__ == "__main__":
    main()
