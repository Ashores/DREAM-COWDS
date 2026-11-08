"""Pure helpers for evaluation provenance and order-correct routing diagnostics."""
import hashlib
from pathlib import Path


def checkpoint_stage_map(payload):
    if payload is None:
        return {}
    stream = payload["config"]["stream"]
    names = [task["name"] for task in stream]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint stream repeats a dataset")
    return {name: index for index, name in enumerate(names)}


def sha256_file(path):
    if path is None:
        return None
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
