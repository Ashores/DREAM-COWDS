# Third-party software and data

DREAM builds on [Meta SAM 3](https://github.com/facebookresearch/sam3).
SAM 3 source code and weights must be obtained separately under the
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
A copy of that upstream license is retained in `licenses/SAM3_LICENSE.txt` for
reference; it does not grant ownership of upstream code or datasets to this project.
This repository does not redistribute SAM 3 model weights or its source tree.

The implementation includes a SAM 3 adaptation of the ideas in
[SAMCL](https://github.com/INV-WZQ/SAMCL), and baseline implementations of LoRA,
LwF, EWC, rehearsal and distillation. The original SAMCL repository is not
vendored here. Comments identifying the source of the selector and shared
projection design are retained in `continual_lora/methods.py`.

PyTorch, NumPy, Pillow, OpenCV and pycocotools are external dependencies and retain
their respective licenses. The five source datasets and SA-Co/Gold retain their
original access conditions and licenses; see `docs/DATASETS.md`.

No blanket license for the authors' original code is granted by this preparation
package. Rights remain with the respective authors and rights holders. An
author-approved software license can be added before public distribution.
