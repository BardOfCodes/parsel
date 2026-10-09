"""Dataset and part-segmented mesh loading."""

DATASETS = ("compat3d", "partnet")


def validate_dataset(dataset):
    """Return a supported dataset name or raise a clear error."""
    if dataset not in DATASETS:
        supported = ", ".join(DATASETS)
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of: {supported}"
        )
    return dataset
