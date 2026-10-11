"""Append tasks to a queue file atomically (readers always see a complete file)."""

import json
import os
from pathlib import Path


def append_tasks(queue, tasks):
    queue = Path(queue)
    existing = queue.read_text() if queue.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    temporary = queue.with_name(f".{queue.name}.{os.getpid()}.tmp")
    temporary.write_text(existing + "".join((t if isinstance(t, str) else json.dumps(t)) + "\n" for t in tasks))
    os.replace(temporary, queue)
