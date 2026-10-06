# CKESwin-MS core code

[中文说明](README_zh.md)

Core implementation accompanying **Reducing Persistent Specimen-Level Errors in
Pterocarpus Wood Identification with Anatomical Knowledge and Chemical Fingerprints**,
aligned to the local KBS v5.25 manuscript.

## Layout

```text
run.py                          Portable command entry point
configs/
  paths.json                    External input/weight locations
  visual.json                   Archived CKESwin training settings
  rf.json                       Paper-selected random forest settings
src/
  train_tri_modal_swin_fusion_v3_2.py  CKESwin architecture, losses and training
  FusionDataset.py              Image/graph/specimen-spectrum pairing
  dograph.py                    Vessel graph construction and cache loading
  create_graph/                 YOLO/SAM adapters and graph geometry
  util/                         Direct data, split, training and evaluation dependencies
  chemistry/                    Extracted RF helpers and portable fit/export adapter
  evaluation/                   Visual evaluation and diagonal score fusion
tests/                          Tests
tools/check_package.py          Package inventory and syntax checks
```

The word “tri-modal” in the main module name refers to **graph, global-image and
regional-image representations**; chemistry is combined separately at the score level.

## Environment

Use a Python environment compatible with PyTorch, torchvision and PyG. Install a
matching torch/torchvision build for your CPU/CUDA platform, then:

```sh
python -m pip install -r requirements.txt
# Only if constructing vessel graphs from raw images:
python -m pip install -r requirements-anatomy.txt
# Only if using the original Bayesian RF search:
python -m pip install -r requirements-search.txt
```

`timm==1.0.22` is the compatibility target used for this package.

## Main commands

```sh
# 1. Optional: construct graphs (requires external detector/SAM weights).
python run.py build-graphs

# 2. Fit the selected RF on non-test specimens and export per-spectrum probabilities.
python run.py train-rf --csv inputs/spectra_20mmu_3percent.csv --test-list inputs/splits/test.txt --output outputs/ms_rf

# 3. Set visual.json rf_csv to outputs/ms_rf/ms_probabilities.csv after checking class order.
python run.py train --config configs/visual.json

# 4. Evaluate the visual checkpoint.
python run.py evaluate --run_dir outputs/ckeswin --weights best.pt

# 5. Fit the 21-parameter head on training + validation pairs, then evaluate held-out pairs.
python run.py fuse --run_dir outputs/ckeswin --weights best.pt --rf_csv outputs/ms_rf/ms_probabilities.csv --stacking_mode diag
```
