# External input contract

No example data or specimen identifiers are distributed.

## Images and graphs

- `IMAGEROOT` contains one directory per species, each with transverse images.
- Class IDs come from lexicographically sorted species directory names.
- `SAVEROOT/processed/data.pt` is the optional PyG graph cache. Building it requires
  the external vessel detector and SAM ViT-H checkpoint. The graph builder expects
  the original detector class convention (`skip_cls=1`); a different detector's
  class labels must not be assumed equivalent.
- The recovered graph implementation has 28 node features and 9 edge features.
- Each graph carries `img_path`, vessel positions, geometry and species label.
  Image folders remain necessary even when a processed cache exists.
- If the cache was produced on another machine, `CKESWIN_ORIGINAL_IMAGE_ROOT` may
  identify its old image-root prefix. `IMAGEROOT` supplies the new prefix. The
  historical `dataset/images/random_test/` suffix is also recognized.

## Specimen splits

Provide `val.txt` and `test.txt` in the same directory. Lines contain a specimen ID,
optionally followed by a comma or tab and a species name. The field `test_file`
in the visual config points to **val.txt**, preserving the recovered convention.
Remaining observations are assigned to training. Specimen sets must not overlap.
IDs are matched within image filenames by the original case-insensitive substring
rule; use unambiguous identifiers. The package does not infer or recreate the
historical specimen lists.

## Chemical features

The RF input CSV has a header and columns in this order:

1. Species label.
2. Specimen ID, matching the held-out list.
3. Metadata column (excluded from model features).
4. Binned intensity columns.

The paper setting expects 1,827 finite numeric features, previously processed at
20 mmu and 3% relative intensity. No raw-spectrum binning/thresholding code was
found. The export adapter rejects missing/nonfinite intensities rather than
silently estimating imputation statistics using held-out observations.

The RF exporter writes `specimen_id`, `species`, `pred_species`, and `p0` ... `p6`
for the seven-class case. It exports every spectrum, retaining repeated specimens.
It also writes `probability_class_order.json`. Check that order against the graph
`class_to_idx.json`; the fusion model does not infer arbitrary class remappings.

The original visual loader chooses a random same-specimen spectrum during
training. For validation/test it averages log probabilities across same-specimen
spectra and applies softmax (a normalized geometric mean). This is **not** a
fixed one-image/one-spectrum row join. The code preserves that behavior.

## Paths and outputs

`configs/paths.json` configures external images, graph cache, detector/SAM weights
and anatomy device. Environment variables with the same names take priority.
`configs/visual.json` additionally configures Swin pretraining, RF CSV, splits and
training output directory. Relative paths use the current working directory.

On Windows, use `num_workers=0` when executing the recovered DataModule: its
graph-standardization transform is a local closure and is unsuitable for spawned
worker pickling. The archived Linux paper configuration keeps `num_workers=3`.

Do not confuse Swin pretraining, trained CKESwin weights and vessel-detector/SAM
weights. They serve different stages and all are external to this distribution.
