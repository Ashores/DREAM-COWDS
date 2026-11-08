#!/usr/bin/env python3
"""Prepare COWDS datasets as COCO-instance + phrase-conditioned records."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


VOC_LABELS = [
    (1, "aeroplane", "airplane"), (2, "bicycle", "bicycle"),
    (3, "bird", "bird"), (4, "boat", "boat"), (5, "bottle", "bottle"),
    (6, "bus", "bus"), (7, "car", "car"), (8, "cat", "cat"),
    (9, "chair", "chair"), (10, "cow", "cow"),
    (11, "diningtable", "dining table"), (12, "dog", "dog"),
    (13, "horse", "horse"), (14, "motorbike", "motorcycle"),
    (15, "person", "person"), (16, "pottedplant", "potted plant"),
    (17, "sheep", "sheep"), (18, "sofa", "sofa"),
    (19, "train", "train"), (20, "tvmonitor", "tv"),
]

CITY_LABELS = {
    "person": (1, "person"), "rider": (2, "rider"),
    "car": (3, "car"), "truck": (4, "truck"), "bus": (5, "bus"),
    "train": (6, "train"), "motorcycle": (7, "motorcycle"),
    "bicycle": (8, "bicycle"),
}

TRASH_CANONICAL = {
    "rov": "underwater ROV", "plant": "underwater plant",
    "animal_fish": "fish", "animal_starfish": "starfish",
    "animal_shells": "shell", "animal_crab": "crab",
    "animal_eel": "eel", "animal_etc": "other underwater animal",
    "trash_clothing": "discarded clothing", "trash_pipe": "discarded pipe",
    "trash_bottle": "discarded bottle", "trash_bag": "discarded bag",
    "trash_snack_wrapper": "snack wrapper", "trash_can": "discarded can",
    "trash_cup": "discarded cup", "trash_container": "discarded container",
    "trash_unknown_instance": "unknown underwater trash",
    "trash_branch": "underwater branch", "trash_wreckage": "wreckage",
    "trash_tarp": "discarded tarp", "trash_rope": "discarded rope",
    "trash_net": "discarded net",
}


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def mask_annotation(mask: np.ndarray) -> tuple[list[float], float, dict]:
    enc = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = [float(x) for x in mask_utils.toBbox(enc)]
    area = float(mask_utils.area(enc))
    enc["counts"] = enc["counts"].decode("ascii")
    return bbox, area, enc


def coco_base(dataset: str, split: str, categories: list[dict]) -> dict:
    return {
        "info": {"description": f"COWDS prepared {dataset} {split}",
                 "format_version": "1.0", "source_dataset": dataset},
        "licenses": [], "images": [], "annotations": [], "categories": categories,
    }


def normalize_coco(src: dict, dataset: str, split: str, image_prefix: str,
                   canonical: dict[str, str] | None = None) -> dict:
    cats = []
    for c in src["categories"]:
        old = c["name"]
        name = canonical.get(old, old.replace("_", " ")) if canonical else old
        cats.append({"id": int(c["id"]), "name": name,
                     "dataset_label": old, "canonical_name": name,
                     "supercategory": c.get("supercategory", "object")})
    out = coco_base(dataset, split, cats)
    for im in src["images"]:
        x = dict(im)
        x["id"] = int(x["id"])
        x["file_name"] = f"{image_prefix}/{Path(x['file_name']).name}"
        x["source_dataset"] = dataset
        x["source_split"] = split
        out["images"].append(x)
    for ann in src["annotations"]:
        x = dict(ann)
        x["id"], x["image_id"], x["category_id"] = map(
            int, (x["id"], x["image_id"], x["category_id"]))
        x["iscrowd"] = int(x.get("iscrowd", x.get("is_crowd", 0)))
        if "bbox" in x:
            x["bbox"] = [float(v) for v in x["bbox"]]
        if "area" in x:
            x["area"] = float(x["area"])
        out["annotations"].append(x)
    return out


def prepare_voc(raw: Path, out: Path) -> list[Path]:
    root = raw / "VOCdevkit" / "VOC2012"
    symlink(root / "JPEGImages", out / "images")
    cats = [{"id": i, "name": canonical, "canonical_name": canonical,
             "dataset_label": original, "supercategory": "object"}
            for i, original, canonical in VOC_LABELS]
    label_to_cat = {i: i for i, _, _ in VOC_LABELS}
    outputs = []
    for split in ("train", "val"):
        ids = (root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()
        coco = coco_base("PASCAL_VOC_2012", split, cats)
        ann_id = 1
        for image_id, stem in enumerate(ids, 1):
            img_path = root / "JPEGImages" / f"{stem}.jpg"
            with Image.open(img_path) as pil:
                w, h = pil.size
            obj = np.asarray(Image.open(root / "SegmentationObject" / f"{stem}.png"))
            cls = np.asarray(Image.open(root / "SegmentationClass" / f"{stem}.png"))
            coco["images"].append({"id": image_id, "file_name": f"images/{stem}.jpg",
                                    "width": w, "height": h,
                                    "source_dataset": "PASCAL_VOC_2012", "source_split": split})
            for instance_id in sorted(int(x) for x in np.unique(obj) if x not in (0, 255)):
                m = obj == instance_id
                vals = cls[m]
                vals = vals[(vals > 0) & (vals < 21)]
                if vals.size == 0:
                    continue
                class_id = int(np.bincount(vals).argmax())
                if class_id not in label_to_cat:
                    continue
                bbox, area, seg = mask_annotation(m)
                if area <= 0:
                    continue
                coco["annotations"].append({"id": ann_id, "image_id": image_id,
                    "category_id": class_id, "bbox": bbox, "area": area,
                    "segmentation": seg, "iscrowd": 0})
                ann_id += 1
        path = out / "annotations" / f"instances_{split}.json"
        dump_json(path, coco); outputs.append(path)
    return outputs


def prepare_hrsid(raw: Path, out: Path) -> list[Path]:
    outputs = []
    mapping = [("train", "train2017.json", "train2017"),
               ("test", "test2017.json", "val2017")]
    for split, annot, folder in mapping:
        symlink(raw / "JPEGImages" / folder, out / "images" / split)
        src = json.load((raw / "annotations" / annot).open())
        coco = normalize_coco(src, "HRSID", split, f"images/{split}")
        coco["categories"][0].update(canonical_name="ship", name="ship", dataset_label="ship")
        # HRSID contains a small number of source boxes that disagree with the
        # polygon annotation. The mask is authoritative for the unified task.
        image_meta = {int(x["id"]): x for x in coco["images"]}
        valid_annotations = []
        rejected = []
        for ann in coco["annotations"]:
            im = image_meta[int(ann["image_id"])]
            encoded = mask_utils.frPyObjects(ann["segmentation"], int(im["height"]), int(im["width"]))
            encoded = mask_utils.merge(encoded)
            ann["bbox"] = [float(x) for x in mask_utils.toBbox(encoded)]
            ann["area"] = float(mask_utils.area(encoded))
            if ann["area"] <= 0 or ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
                rejected.append({"annotation_id": ann["id"], "image_id": ann["image_id"],
                                 "reason": "degenerate source polygon"})
            else:
                valid_annotations.append(ann)
        coco["annotations"] = valid_annotations
        dump_json(out / "audit" / f"rejected_{split}.json", rejected)
        path = out / "annotations" / f"instances_{split}.json"
        dump_json(path, coco); outputs.append(path)
    return outputs


def prepare_kvasir(raw: Path, out: Path) -> list[Path]:
    image_dir, mask_dir = raw / "images" / "images", raw / "masks" / "masks"
    symlink(image_dir, out / "images" / "all")
    boxes = json.load((raw / "bboxes.json").open())
    stems = sorted(p.stem for p in image_dir.glob("*.jpg"))
    rng = random.Random(3407); shuffled = stems[:]; rng.shuffle(shuffled)
    n_val = round(len(shuffled) * 0.2)
    split_ids = {"all": stems, "train": sorted(shuffled[n_val:]), "val": sorted(shuffled[:n_val])}
    categories = [{"id": 1, "name": "endoscopic instrument",
                   "canonical_name": "endoscopic instrument",
                   "dataset_label": "instrument", "supercategory": "medical instrument"}]
    all_images, all_anns = {}, defaultdict(list)
    ann_id = 1
    for image_id, stem in enumerate(stems, 1):
        p = image_dir / f"{stem}.jpg"
        with Image.open(p) as pil: w, h = pil.size
        all_images[stem] = {"id": image_id, "file_name": f"images/all/{stem}.jpg",
                            "width": w, "height": h, "source_dataset": "Kvasir-Instrument"}
        mask_path = mask_dir / f"{stem}.png"
        union = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if union is None: raise FileNotFoundError(mask_path)
        union = union > 0
        entry = boxes.get(stem)
        if entry is None: raise KeyError(f"Missing Kvasir bbox entry: {stem}")
        for b in entry.get("bbox", []):
            x1=max(0,int(b["xmin"])); y1=max(0,int(b["ymin"])); x2=min(w,int(b["xmax"])); y2=min(h,int(b["ymax"]))
            m=np.zeros((h,w),dtype=bool); m[y1:y2+1,x1:x2+1]=union[y1:y2+1,x1:x2+1]
            if not m.any():
                m[y1:y2+1,x1:x2+1]=True
            bbox, area, seg = mask_annotation(m)
            all_anns[stem].append({"id": ann_id, "image_id": image_id, "category_id": 1,
                "bbox": bbox, "area": area, "segmentation": seg, "iscrowd": 0})
            ann_id += 1
    outputs=[]
    for split, ids in split_ids.items():
        coco=coco_base("Kvasir-Instrument",split,categories)
        allowed=set(ids)
        coco["images"]=[dict(all_images[s],source_split=split) for s in ids]
        coco["annotations"]=[a for s in ids for a in all_anns[s]]
        path=out/"annotations"/f"instances_{split}.json"; dump_json(path,coco); outputs.append(path)
    dump_json(out/"split_manifest.json", {"policy":"fixed 80/20 image split", "seed":3407,
              "train":split_ids["train"],"val":split_ids["val"]})
    return outputs


def prepare_cityscapes(raw: Path, out: Path) -> list[Path]:
    outputs=[]
    categories=[{"id":v[0],"name":v[1],"canonical_name":v[1],"dataset_label":k,"supercategory":"thing"}
                for k,v in CITY_LABELS.items()]
    for split in ("train","val"):
        symlink(raw/"leftImg8bit"/split,out/"images"/split)
        coco=coco_base("Cityscapes",split,categories); ann_id=1
        files=sorted((raw/"gtFine"/split).rglob("*_gtFine_polygons.json"))
        for image_id,jp in enumerate(files,1):
            d=json.load(jp.open()); stem=jp.name.replace("_gtFine_polygons.json","")
            city=jp.parent.name; fn=f"images/{split}/{city}/{stem}_leftImg8bit.png"
            coco["images"].append({"id":image_id,"file_name":fn,"width":d["imgWidth"],
                "height":d["imgHeight"],"source_dataset":"Cityscapes","source_split":split})
            for obj in d["objects"]:
                label=obj["label"]
                if label not in CITY_LABELS or obj.get("deleted",0): continue
                poly=np.asarray(obj["polygon"],dtype=np.float64)
                if len(poly)<3: continue
                x1,y1=poly.min(axis=0); x2,y2=poly.max(axis=0)
                area=float(abs(cv2.contourArea(poly.astype(np.float32))))
                if area<=0: continue
                coco["annotations"].append({"id":ann_id,"image_id":image_id,
                    "category_id":CITY_LABELS[label][0],"bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                    "area":area,"segmentation":[poly.reshape(-1).astype(float).tolist()],"iscrowd":0})
                ann_id+=1
        path=out/"annotations"/f"instances_{split}.json"; dump_json(path,coco); outputs.append(path)
    return outputs


def prepare_trashcan(raw: Path, out: Path) -> list[Path]:
    outputs=[]
    for split in ("train","val"):
        symlink(raw/split,out/"images"/split)
        src=json.load((raw/f"instances_{split}_trashcan.json").open())
        coco=normalize_coco(src,"TrashCan-Instance",split,f"images/{split}",TRASH_CANONICAL)
        path=out/"annotations"/f"instances_{split}.json"; dump_json(path,coco); outputs.append(path)
    return outputs


def phrase_files(coco_path: Path, dataset_out: Path) -> dict:
    data=json.load(coco_path.open()); split=coco_path.stem.replace("instances_","")
    cats={int(c["id"]):c for c in data["categories"]}
    grouped=defaultdict(list)
    for a in data["annotations"]: grouped[(int(a["image_id"]),int(a["category_id"]))].append(a)
    present=defaultdict(set)
    for iid,cid in grouped: present[iid].add(cid)
    phrase_dir=dataset_out/"phrases"; phrase_dir.mkdir(parents=True,exist_ok=True)
    pos_path=phrase_dir/f"positive_{split}.jsonl"; neg_path=phrase_dir/f"negative_candidates_{split}.jsonl"
    npos=nneg=0
    with pos_path.open("w",encoding="utf-8") as pos, neg_path.open("w",encoding="utf-8") as neg:
        for im in data["images"]:
            iid=int(im["id"])
            for cid in sorted(present[iid]):
                c=cats[cid]; anns=grouped[(iid,cid)]
                rec={"image_id":iid,"file_name":im["file_name"],"prompt":c["canonical_name"],
                    "category_id":cid,"presence":1,
                    "instances":[{"annotation_id":a["id"],"bbox":a["bbox"],"segmentation":a["segmentation"],"area":a["area"]} for a in anns],
                    "source":{"dataset":data["info"]["source_dataset"],"split":split}}
                pos.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n"); npos+=1
            for cid in sorted(set(cats)-present[iid]):
                c=cats[cid]
                rec={"image_id":iid,"file_name":im["file_name"],"prompt":c["canonical_name"],
                     "category_id":cid,"presence":0,"instances":[],
                     "source":{"dataset":data["info"]["source_dataset"],"split":split}}
                neg.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n"); nneg+=1
    return {"split":split,"images":len(data["images"]),"instances":len(data["annotations"]),
            "categories":len(cats),"positive_phrases":npos,"negative_candidates":nneg}


def validate(coco_path: Path, dataset_out: Path) -> list[str]:
    d=json.load(coco_path.open()); errors=[]
    image_ids={int(x["id"]) for x in d["images"]}; cat_ids={int(x["id"]) for x in d["categories"]}
    if len(image_ids)!=len(d["images"]): errors.append("duplicate image ids")
    ann_ids=set()
    for a in d["annotations"]:
        if a["id"] in ann_ids: errors.append(f"duplicate annotation id {a['id']}")
        ann_ids.add(a["id"])
        if int(a["image_id"]) not in image_ids: errors.append(f"bad image ref {a['id']}")
        if int(a["category_id"]) not in cat_ids: errors.append(f"bad category ref {a['id']}")
        x,y,w,h=map(float,a["bbox"])
        if w<=0 or h<=0 or float(a.get("area",0))<=0: errors.append(f"invalid geometry {a['id']}")
    for im in d["images"]:
        if not (dataset_out/im["file_name"]).exists(): errors.append(f"missing image {im['file_name']}")
    return errors[:100]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); args=ap.parse_args()
    raw=args.root/"datasets"; prepared=args.root/"prepared_datasets"
    prepared.mkdir(parents=True,exist_ok=True)
    jobs=[("PASCAL_VOC_2012",prepare_voc,raw/"PASCAL_VOC_2012"),
          ("HRSID",prepare_hrsid,raw/"HRSID-JPG"),
          ("Kvasir-Instrument",prepare_kvasir,raw/"Kvasir-Instrument"),
          ("Cityscapes",prepare_cityscapes,raw/"CityScapes"),
          ("TrashCan-Instance",prepare_trashcan,raw/"TrashCan-Instance")]
    manifest={"format_version":"1.0","format":"COCO Instance + phrase-conditioned JSONL",
              "task_order":[x[0] for x in jobs],"datasets":{}}
    for name,fn,src in jobs:
        dst=prepared/name; dst.mkdir(parents=True,exist_ok=True)
        paths=fn(src,dst); stats=[]; errors=[]
        for path in paths:
            stats.append(phrase_files(path,dst)); errors.extend(validate(path,dst))
        vocab=json.load(paths[0].open())["categories"]
        dump_json(dst/"vocabulary.json",vocab)
        manifest["datasets"][name]={"path":str(dst),"splits":stats,"validation_errors":errors}
        print(name,stats,"errors",len(errors),flush=True)
        if errors: raise RuntimeError(f"Validation failed for {name}: {errors[:5]}")
    dump_json(prepared/"manifest.json",manifest)
    print("COMPLETE",prepared)


if __name__ == "__main__":
    main()
