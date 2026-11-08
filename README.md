# DREAM on COWDS

Implementation accompanying **DREAM: Decoupled Routing Expert and Accumulated
Memory for Continual Open-Vocabulary Detection and Segmentation**.

DREAM adapts a frozen SAM 3 backbone with a shared visual LoRA, routed private
adapters, and covariance-guided query subspace memory (CQSM). The supplied
configuration is the final **covariance basis, memory rank 2, routed access**
configuration. It uses direct addition of the Global branch, without the learned
Global/Private deconflicting operator.

The code contains training, full checkpoint evaluation, dataset conversion,
the exact Kvasir image split inventory, and CPU mechanism tests. Source images,
SAM 3 weights and trained DREAM checkpoints are not bundled.

## Installation

Use a Linux CUDA environment and install the original SAM 3 model and training
dependencies following [Meta's installation instructions](https://github.com/facebookresearch/sam3)
and [training guide](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md).
Obtain the original SAM 3 checkpoint through the upstream access process. This
recipe was developed for SAM 3, not SAM 3.1. Record the upstream revision and
environment used for a reproduction.

After the SAM 3 training installation is working, install this package's direct
dependencies from this repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

`requirements.txt` lists direct requirements; it is not a reconstructed lockfile
of the original GPU environment. Local release checks used Python 3.11.7,
PyTorch 2.9.1 (CPU), NumPy 2.2.6 and pytest 7.4.0. They verify the adapter tests,
not a new GPU training run or current upstream SAM 3 compatibility.

## Data preparation

Download the five source datasets from the providers listed in
[docs/DATASETS.md](docs/DATASETS.md), respecting their access conditions.
Arrange the raw files as documented there, then run:

```bash
python prepare_cowds_datasets.py --root /path/to/cowds_data
python tools/build_clean_data_view.py \
  --source-root /path/to/cowds_data/prepared_datasets \
  --output-root /path/to/cowds_data/evaluation_view
```

Use `evaluation_view` as `--data-root` for both training and evaluation. The
view keeps the training samples unchanged and reproduces the evaluated Kvasir
validation set. The builder checks source annotation hashes and all Kvasir image
hashes before writing. It fails rather than silently substituting another split.

Prepared datasets use COCO instance annotations and category prompts. The exact
split sizes, annotation hashes and Kvasir membership are in `data_manifests/`.
The converter writes the category names used as prompts to each dataset's
`vocabulary.json`.

## Train the final configuration

The recorded training recipe uses two GPUs, one image per GPU and 1,000 updates
per task. Set the paths to your SAM 3 checkpoint, tokenizer vocabulary, and
prepared data. `--dry-run` prints the command without loading a model or using a
GPU.

```bash
python tools/run_dream.py train \
  --data-root /path/to/cowds_data/evaluation_view \
  --checkpoint /path/to/sam3.pt \
  --bpe-path /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --output-dir outputs/dream_seed42 --gpus 0,1 --seed 42
```

`configs/dream_routed.json` records all training options. Private rank is 16,
Global rank is 10, CQSM rank is 2, and both learning rates are `1e-4`. The
default order is VOC, HRSID, Kvasir, Cityscapes and TrashCan. Change the
`--stream-order` value in a copied configuration to evaluate another permutation;
every dataset must appear once. Use `--config` to select that file.

Training produces `config.json`, `history.json` and five `task_*.pt` adapter
checkpoints. The adapters require the original SAM 3 backbone at evaluation.
For the historical seed-42 final checkpoint, the recorded SHA256 is
`38d17fd3aaf0c881b7912b675948d4c25fe66ea8d115a1d1fd17aac35a5bd357`.
This checksum identifies that checkpoint; a new training run need not produce
identical serialized bytes.

## Evaluate the complete stream

First compute the frozen SAM 3 references on the same prepared annotations:

```bash
python tools/run_dream.py frozen \
  --data-root /path/to/cowds_data/evaluation_view \
  --checkpoint /path/to/sam3.pt \
  --bpe-path /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --output-dir outputs/frozen --gpus 0,1
```

Then evaluate every checkpoint on all five domains:

```bash
python tools/run_dream.py evaluate \
  --data-root /path/to/cowds_data/evaluation_view \
  --checkpoint /path/to/sam3.pt \
  --bpe-path /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --method-dir outputs/dream_seed42 --frozen-root outputs/frozen \
  --output-dir outputs/dream_seed42_eval --gpus 0,1
```

This schedules 25 checkpoint/domain evaluations, with one worker per selected
GPU, and writes `continual_metrics.json`. Box and mask AP values in JSON use
the `[0,1]` scale; multiply by 100 for AP points. The default prediction
threshold is 0.3. Cached adapter results and frozen references must match the
SAM 3 weight hash, adapter identity, annotation hash, image count and threshold.
Legacy results without the SAM 3 hash require a new evaluation directory.
Retain all five checkpoints
to measure forgetting and the full learning trajectory.

For predictions from a single checkpoint, call `frozen_sam3_baseline.py` with
`--adapter-checkpoint` and `--keep-predictions`; see its `--help` output. The
ordinary evaluation selects the route from the image. Do not enable the optional
oracle-routing diagnostic flags for the paper's task-free evaluation.

## Tests and source provenance

```bash
python -m pytest -q test_cqsm.py test_cqsm_restored.py \
  test_global_full_private.py test_shared_prefix.py test_numerics.py \
  test_release_provenance.py
```

The 25 experiment-source files were reconstructed against their recorded SHA256
manifest. `SOURCE_PROVENANCE.json` records their original and release hashes.
Three existing entry points have release-only changes: explicit portable path
arguments, selection of the current Python interpreter, and validation of cached
results against SAM 3 weight hashes and expected image coverage. Predictions and
metric formulas are unchanged. Older comparison implementations remain in the shared
module to preserve checkpoint loading and test dependencies; the supplied
configuration selects DREAM explicitly.

The repository currently covers the five-domain COWDS train/evaluate workflow.
The separately executed SA-Co/Gold evaluation and profiling pipelines are not
included in this initial code package. No trained weights are claimed to be
downloadable here.

## Attribution and use

Please cite the DREAM paper and the original SAM 3 and dataset papers when using
this benchmark. Upstream software and datasets retain their own terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); an author-approved license for
the original project code has not yet been assigned.
