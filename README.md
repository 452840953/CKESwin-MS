# CKESwin-MS core code

[中文说明](README_zh.md)

Core implementation accompanying **Reducing Persistent Specimen-Level Errors in
Pterocarpus Wood Identification with Anatomical Knowledge and Chemical Fingerprints**,
aligned to the local KBS v5.24 manuscript.

This distribution contains source code, configuration templates, documentation and
synthetic tests only. It contains **no datasets, specimen lists, model weights,
predictions, experiment logs or research results**. It is a curated core-code
extraction, not the full archive of baselines, ablations and figure-generation work.

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
tests/                          Synthetic checks; no research data
tools/check_package.py          Source-only inventory and syntax checks
docs/                           Paper mapping, changes, input contract and provenance
```

The historical main module name is retained for dynamic imports and checkpoint
compatibility. Its word “tri-modal” refers to **graph, global-image and regional
image representations**; chemistry is combined separately at the score level.

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

The requirements are inferred direct dependencies; they are **not** a historical
environment lock. `timm==1.0.22` is the compatibility target used for this package.

## External inputs

Edit `configs/paths.json` and `configs/visual.json`. Run commands from the repository
root; relative paths resolve against the working directory. Environment variables
override the anatomy paths. No inputs are supplied by this code distribution.

See [the input contract](docs/INPUTS.md) before running. In particular:

- Images must be grouped into species directories. Graph class order and RF
  probability-column order must agree.
- The legacy `test_file` configuration field actually names **val.txt**. The true
  held-out list must be named **test.txt in the same directory**.
- Raw DART spectra must already have been binned/thresholded externally. The raw
  binning implementation was not found and is not reconstructed here.
- Existing graph caches may contain old image paths; use `IMAGEROOT` and, if needed,
  `CKESWIN_ORIGINAL_IMAGE_ROOT` to remap them.
- Swin pretraining and YOLO/SAM checkpoints must be supplied externally. Imports
  and synthetic tests do not load detector weights or download pretrained models.

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

RF fitting defaults to the paper-selected hyperparameters (1,000 trees, depth 26).
`--search bayes` or `--search random` selects the respective original RF search
space, using five specimen-grouped 75:25 splits of the non-test pool. The held-out
specimens are excluded from fitting/search. The fixed route does not repeat model
selection. A later run will generate outputs and weights under `outputs/`; none
are bundled in this source-only distribution.

```sh
python tools/check_package.py
python -B tests/test_core.py -v
python -B tests/test_model_smoke.py -v
python -B tests/test_rf_adapter.py -v
```

See [paper correspondence and limitations](docs/PAPER_MAPPING.md) and
[packaging changes](docs/CHANGES.md). Synthetic tests are not a reproduction of
the paper's trained-model results. The recovered readable source is newer than
the historical run; unchanged architecture/loss code and recovered settings do
not establish bit-for-bit historical equivalence.

## Publication metadata

The code is released under the MIT License, copyright 2026 zwh. The publication
DOI has not yet been supplied, so no DOI is stated here. Third-party model weights
are not distributed; users must follow the licenses and notices applicable to
their own external dependencies and checkpoints.
