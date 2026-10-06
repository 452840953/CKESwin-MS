# Packaging validation

Date: 2026-10-06. These are software checks, not experiments on the research data.

- All packaged Python files parse successfully.
- Static local-import inspection found no missing project modules.
- All 22 top-level architecture/loss/helper definitions in the main model module
  remain AST-identical to the source; only `TrainConfig` and `main` changed.
- Six core synthetic tests passed: paper configuration, Gaussian geometry,
  GINE/Transformer output shapes, the 21-parameter fusion formula and gradients,
  specimen-group holdout, and detector-free imports.
- One full-model synthetic CPU forward test passed: seven-class output, four
  expected Swin map sizes, three 256-D embeddings, unique parameter and checkpoint
  tensor counts. Its checkpoint-loader input was the random backbone state held
  in memory; no historical checkpoint was tested or used.
- One RF CLI integration test passed using generated toy inputs: fitting,
  probability export, label order and exclusion of held-out specimen groups.
  Toy inputs and outputs were confined to an automatically removed temporary
  directory outside this package.

Core/full-model checks used Python 3.9.23, torch 1.13.1+cu116 on CPU,
torchvision 0.14.1+cu116, torch-geometric 2.6.1 and timm 1.0.22. RF adapter
integration was also checked under the available Python 3.11 environment.
These are validation environments, not a recovered historical training lock.

Not performed: 300-epoch retraining, YOLO/SAM inference, evaluation on research
data, historical checkpoint compatibility, paper-metric reproduction or multi-GPU
training. External data and weights are intentionally absent.

Run `python tools/check_package.py` to recheck the final source-only inventory and
the extraction hashes. Run the three test scripts under `tests/` for synthetic
runtime checks. Use `-B` to keep Python bytecode caches out of the source package.
