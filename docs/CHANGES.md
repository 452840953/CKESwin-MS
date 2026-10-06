# Extraction and packaging changes

The original archive and manuscript files were not modified.

- Extracted the selected CKESwin module and its local import dependency closure.
  Other training versions, test sweeps, notebooks, weights, data and research
  outputs were excluded.
- Pruned shared utility modules to functions needed by the selected chain. Removed
  a legacy post-training plotting call that expected an older gated-model API;
  use `run.py evaluate` instead. The training loop and checkpoint selection remain
  unchanged.
- Moved the two formal evaluation entries into `src/evaluation/`; retained the
  historical main model module name for dynamic imports.
- Restored epochs and auxiliary annealing defaults from the archived paper run:
  300 epochs, warm-up 10, decay 100, minimum auxiliary factor 0.1. Architecture,
  forward computation and loss functions were not rewritten.
- Added an optional configuration argument to the original training entry and a
  small root CLI. Replaced machine-specific paths with external-input templates.
- Replaced the hard-coded image-path rewrite with configurable root remapping.
- Deferred YOLO/SAM weight loading until graph construction. Importing the model
  no longer requires detector checkpoints. Optional detector libraries are also
  imported lazily.
- Changed import-time logging to console by default. Set `CKESWIN_FILE_LOGS=1`
  to explicitly enable file logs during execution.
- Deferred evaluation of union type annotations in `fusion_debug.py` so imports
  also work with Python 3.9; no numerical computation changed.
- Visual/fused test evaluation requires a nonempty held-out split rather than
  falling back to validation. Fusion also requires both train and validation
  pairs, matching the selected paper protocol. The CLI permits only the diagonal
  fusion mode; the inherited alternative class is not an advertised paper method.
- Applied the RF CSV override before DataModule setup so the command-line path
  actually governs loaded probabilities.
- Extracted only the required RF helper functions from the local mass-spectrometry
  archive. Added a clearly identified portable RF fitting/export adapter and
  fixed paper hyperparameters. No all-model comparison or exploratory XGBoost
  training script was copied.
- Added synthetic checks and a source-only inventory validator. Tests neither
  read research inputs nor write trained checkpoints or predictions to this package.

`source_manifest.json` records original and packaged hashes for extracted files.
Root labels identify the original visual code tree and the separate local MS
archive without embedding a user's absolute machine paths. Files not listed as
extractions (CLI, docs, configs, tests and the RF adapter) are new packaging files.
