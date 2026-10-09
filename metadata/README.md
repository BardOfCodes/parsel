# Evaluation metadata

These CSV files replace the legacy pickle files used to select the PartNet
and Compat3D evaluation edits. They are plain-text release artifacts and do
not contain mesh geometry.

## PartNet

`partnet_edits.csv` contains 52 edit requests for 47 unique PartNet shapes.
Its rows come from:

- edit selection: `dataset_mod.pkl`
  (`sha256:69c824179305afd241227f57b8fcef9d8b13fdfb6557fa803a6ff0fa1e7eb6e1`);
- category and ShapeNet mapping: `class_wise_dict.pkl`
  (`sha256:7d1ea3f683a66365770be1e4d8a8ff9d7c080e038c738b254b96947e7e7e1078`).

Columns:

- `row_index`: position in the legacy list, which is what the old scripts
  actually used to select a request;
- `dataset_index`: index stored inside that row;
- `shape_id`: PartNet annotation ID;
- `category`: PartNet category, joined from the catalog;
- `shapenet_model_id`: corresponding ShapeNet model ID;
- `edit_request`: natural-language edit.

The two index columns intentionally differ at rows 47 and 51. The source
pickle stores `dataset_index=51` at row 47 and `dataset_index=47` at row 51.
Both values are preserved so this inconsistency is visible instead of being
silently repaired.

## Compat3D

`compat3d_edits.csv` contains 50 edit requests for 50 unique Compat3D shapes.
It comes from `compat_edits_subset.pkl`
(`sha256:4378b0d5a0175192563a6517b50caf0a0a5ce4507d732cfcdac4aa887e9e6d67`).

The aggregate `total` row from the pickle is excluded. Source headers were
normalized to snake case, and the original `type` field is named
`request_type`. The blank `keep_fixed` value for shape `7674` is preserved as
missing data.

## Regeneration

Only unpickle trusted files. To regenerate or verify the CSVs:

```bash
python scripts/convert_metadata.py \
    --partnet-edits /path/to/dataset_mod.pkl \
    --partnet-catalog /path/to/class_wise_dict.pkl \
    --compat3d-edits /path/to/compat_edits_subset.pkl

python scripts/convert_metadata.py \
    --partnet-edits /path/to/dataset_mod.pkl \
    --partnet-catalog /path/to/class_wise_dict.pkl \
    --compat3d-edits /path/to/compat_edits_subset.pkl \
    --check
```
