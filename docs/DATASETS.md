# Datasets, prompts and evaluated splits

COWDS uses existing datasets. Download raw files from their providers; no source
images are redistributed in this repository.

| Dataset | Provider | Train / evaluation images | Categories |
|---|---|---:|---:|
| PASCAL VOC 2012 | [Oxford VOC project](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/) | 1,464 / 1,449 | 20 |
| HRSID | [Authors' repository](https://github.com/chaozhong2010/HRSID) | 3,642 / 1,962 | 1 |
| Kvasir-Instrument | [Simula dataset page](https://datasets.simula.no/kvasir-instrument/) | 472 / 115 | 1 |
| Cityscapes | [Dataset website](https://www.cityscapes-dataset.com/) | 2,975 / 500 | 8 |
| TrashCan-Instance | [University of Minnesota IRV Lab](https://irvlab.cs.umn.edu/resources/trashcan) | 6,065 / 1,147 | 22 |

The recorded evaluation splits are VOC `val`, HRSID `test`, Kvasir `val`,
Cityscapes `val` and TrashCan-Instance `val`. VOC uses the segmentation subset.
HRSID boxes are derived from the instance polygons and degenerate polygons are
excluded. Cityscapes uses its eight instance categories; crowd/group regions are
excluded by the converter.

For this study, Kvasir was partitioned with a fixed 80/20 shuffle using seed 3407.
Three validation images that are byte-identical to training images were removed,
leaving 115 validation images and 121 instances. Training retains 472 images and
496 instances. `data_manifests/kvasir_image_inventory.json` gives the exact
membership and image hashes for train, original validation and evaluated
validation. The removal is specific to this recorded protocol; do not assume it
is the provider's current default split.

## Raw layout

The converter expects these extracted layouts under the directory passed as
`prepare_cowds_datasets.py --root`:

```text
cowds_data/
  datasets/
    PASCAL_VOC_2012/VOCdevkit/VOC2012/
      JPEGImages/ SegmentationObject/ SegmentationClass/ ImageSets/Segmentation/
    HRSID-JPG/
      JPEGImages/train2017/ JPEGImages/val2017/
      annotations/train2017.json annotations/test2017.json
    Kvasir-Instrument/
      images/images/ masks/masks/ bboxes.json
    CityScapes/
      leftImg8bit/train/ leftImg8bit/val/ gtFine/train/ gtFine/val/
    TrashCan-Instance/
      train/ val/ instances_train_trashcan.json instances_val_trashcan.json
```

Use the matching original archive contents. The exact validation hashes provide
a check against accidental changes in annotation versions. If the files fail the
hash check, inspect the source version and conversion output instead of disabling
the check. Creating the data view uses symbolic links; Linux is recommended.

## Prepared layout and prompts

```text
prepared_datasets/<dataset>/
  images/...
  annotations/instances_train.json
  annotations/instances_<eval-split>.json
  vocabulary.json
  phrases/positive_<split>.jsonl
  phrases/negative_candidates_<split>.jsonl
```

Instance annotations use COCO boxes `[x,y,width,height]` and masks encoded as
polygons or RLE. Each image path is relative to its dataset directory. Category
names are the text prompts used by training and evaluation. `vocabulary.json`
also retains the original dataset labels. HRSID uses `ship`; Kvasir uses
`endoscopic instrument`; the multi-category datasets use the canonical category
mapping in `prepare_cowds_datasets.py`.

The clean evaluation view deliberately exposes only the applicable Kvasir
annotations, image links, vocabulary and split manifest. Its training annotations
are unchanged. Dataset identities in provenance records are not additional
inference inputs.

SA-Co/Gold is an external evaluation resource from
[the SAM 3 authors](https://github.com/facebookresearch/sam3).
Its original multi-annotator files and official evaluation protocol should be
obtained from that source; the initial COWDS code package does not bundle its
separate evaluator.
