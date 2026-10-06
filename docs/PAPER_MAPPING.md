# Paper-to-code correspondence

Reference: English and Chinese KBS v5.24 manuscripts and the associated
supplementary methods. This is a source-level correspondence record, not a claim
that trained-model results were reproduced.

| Paper component | Implementation | Selection evidence |
|---|---|---|
| YOLO boxes + SAM ViT-H vessel masks | `dograph.py`, `create_graph/vision/` | Main graph dataset dependency chain |
| Vessel geometry and pairwise relations | `util/graph_create.py`, `create_graph/graph/` | Main graph dataset dependency chain |
| Four-layer GINE, hidden width 128 | `GraphEncoder` in the main model module | Architecture and archived visual config |
| Four Swin feature maps; radius-aware Gaussian pooling | `SwinEncoder`, `gaussian_weight_map`, `TriModalSwinFusionModel` | Main module selected by archived fusion config |
| 256-D graph/global/region tokens; two-layer four-head Transformer | `TriModalFusionTransformer` | Architecture and archived visual config |
| Auxiliary classification, reconstructions and alignment | `run_epoch`, `contrastive_nt_xent` | Original loss implementation retained |
| 300 epochs, archived auxiliary-loss annealing | `configs/visual.json` | Archived `ckeswin_cfg.json`, not later script defaults |
| Specimen-disjoint visual partitions | `util/split_util.py` | Original val/test list convention retained |
| RF fitting on non-test specimens | `chemistry/rf_helpers.py`, `chemistry/train_rf.py` | Helpers extracted from archived `machLearn2.py`; wrapper explicitly identified as new |
| RF: 1000 trees, depth 26, sqrt features | `configs/rf.json` | Archived selected RF parameter record |
| 21-parameter classwise score fusion | `evaluation/fusion.py: DiagStackingHead` | Same source as the current manuscript's experimental-lineage copy |
| Fusion fitting on training + validation pairs | `evaluation/fusion.py: main` | Actual concatenation in source and current Supplementary Methods S4 |

## Matters that remain qualified

1. **Historical version:** The recovered readable main source is later than the
   archived principal run. The old package notes this version drift. Architecture
   and loss implementations are preserved; known training-default drift is
   corrected from the saved configuration. Historical binary equivalence remains
   unestablished. No bytecode, checkpoints or results are shipped as substitutes.
2. **Raw spectral preprocessing:** The original raw-spectrum binning implementation
   was not located. RF code starts with the binned table. A generic replacement
   was not invented and labelled as the original method.
3. **“Triplet alignment” wording:** The source computes three pairwise symmetric
   NT-Xent alignment terms among graph/global/region embeddings. It does not use
   a conventional margin-based `TripletMarginLoss`. This terminology distinction
   should be retained when describing the code.
4. **Multiple spectra:** Training uses random same-specimen spectrum selection;
   validation/test uses normalized geometric-mean probabilities. “Paired” in the
   manuscript should not be interpreted as a fixed spectrum row for every image.
5. **Hardware:** The recovered main entry selects one GPU; no DataParallel/DDP
   wrapper is present in that main path. A record of two available RTX 3090 GPUs
   does not itself establish two-GPU parallel training. No parallel-training code
   was added to force agreement with the paper wording.
6. **Specimen counts:** The package neither ships nor reconstructs specimen lists.
   Archived sources distinguish the 46-specimen chemical training-side pool from
   the smaller set with usable visual pairs. Verify actual split inputs rather
   than hard-coding manuscript counts into the loader.
7. **RF fitting adapter:** Fixed-parameter fitting and probability export are new
   packaging entry points built around the archived split/preprocessing helpers
   and selected RF parameters. They are not claimed to be a recovered historical
   probability-export script. The RF search spaces are transcribed from the
   original RF branch; other candidate classifiers are excluded from this core
   distribution.
8. **Scope:** Baseline models, ablations, CAM/ROI analyses, specimen-level
   statistical reports and paper figures are outside this core-method extraction.
   This package is not a complete reproduction archive for every paper table.
9. **Parameter-count convention:** With 28 node features and 9 edge features, the
   recovered model contains 95,724,822 parameter elements plus 2,026 persistent buffer
   elements (BatchNorm statistics/counters and GINE epsilon buffers). Their sum
   is 95,726,848, the manuscript's unique checkpoint-tensor count. The latter
   should not be described as a count of trainable parameters alone.
   Non-persistent Swin attention buffers are excluded from the checkpoint count.

The original old-package note that the head was fitted only on validation data is
not adopted: the current selected fusion source explicitly concatenates training
and validation logits, as described in v5.24.
