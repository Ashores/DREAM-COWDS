"""Recreate the evaluated data view without changing any source image or annotation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    source, output = a.source_root.resolve(), a.output_root.resolve()
    if output == source or output.exists(): p.error("Output must be a new, separate directory")
    protocol = json.loads((ROOT / "data_manifests/evaluation_protocol.json").read_text())
    inventory = json.loads((ROOT / "data_manifests/kvasir_image_inventory.json").read_text())
    for relative, expected in protocol["source_annotation_sha256"].items():
        if sha(source / relative) != expected:
            raise ValueError(f"Annotation differs from the recorded source: {relative}")
    dataset = source / "Kvasir-Instrument"
    train = json.loads((dataset / "annotations/instances_train.json").read_text())
    val = json.loads((dataset / "annotations/instances_val.json").read_text())
    for split, record in [("train", train), ("val_original", val)]:
        expected = {(r["image_id"], r["file_name"]): r["sha256"] for r in inventory[split]}
        actual = {(im["id"], im["file_name"]): sha(dataset / im["file_name"])
                  for im in record["images"]}
        if actual != expected: raise ValueError(f"Kvasir image inventory mismatch: {split}")
    remove = set(protocol["kvasir_removed_val_image_ids"])
    clean = dict(val)
    clean["images"] = [im for im in val["images"] if im["id"] not in remove]
    clean["annotations"] = [ann for ann in val["annotations"] if ann["image_id"] not in remove]
    expected_val = protocol["datasets"]["Kvasir-Instrument"]["val"]
    assert len(clean["images"]) == expected_val["images"] == 115
    assert len(clean["annotations"]) == expected_val["annotations"] == 121
    payload = (json.dumps(clean, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == expected_val["sha256"]
    if a.check_only:
        print("Source annotations and all Kvasir images match the recorded protocol.")
        return
    output.mkdir(parents=True)
    for name in protocol["datasets"]:
        if name != "Kvasir-Instrument": (output / name).symlink_to(source / name, target_is_directory=True)
    target = output / "Kvasir-Instrument"
    (target / "annotations").mkdir(parents=True)
    (target / "images").symlink_to(dataset / "images", target_is_directory=True)
    (target / "vocabulary.json").symlink_to(dataset / "vocabulary.json")
    (target / "annotations/instances_train.json").symlink_to(dataset / "annotations/instances_train.json")
    (target / "annotations/instances_val.json").write_bytes(payload)
    write(target / "split_manifest.json", {
        "policy": "Fixed 80/20 partition; exact-byte train/val duplicates removed from val",
        "seed": 3407,
        "train": [Path(r["file_name"]).stem for r in inventory["train"]],
        "val": [Path(r["file_name"]).stem for r in inventory["val_clean"]],
        "excluded_val_image_ids": sorted(remove)})
    write(output / "protocol.json", protocol)
    print(f"Created evaluated data view: {output}")


if __name__ == "__main__": main()
