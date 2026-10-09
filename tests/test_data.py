import importlib.util
import pickle

import pytest

from parsel.data import DATASETS, validate_dataset
from parsel.data.loading import load_shape


def test_dataset_names_are_explicit_and_validated():
    assert DATASETS == ("compat3d", "partnet")
    assert validate_dataset("partnet") == "partnet"
    assert validate_dataset("compat3d") == "compat3d"

    with pytest.raises(ValueError, match="Unsupported dataset"):
        validate_dataset("compat")


def test_shared_loader_has_a_dataset_neutral_module_name():
    assert importlib.util.find_spec("parsel.data.loading") is not None
    assert importlib.util.find_spec("parsel.data.partnet") is None


def test_missing_cache_does_not_trigger_implicit_regeneration(tmp_path):
    cache_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="regenerate=True"):
        load_shape(
            cache_path,
            tmp_path,
            dataset="partnet",
        )

    assert not cache_path.exists()


@pytest.mark.parametrize(
    "contents",
    [
        b"not a pickle",
        pickle.dumps(("only", "three", "items")),
    ],
)
def test_invalid_cache_is_not_rewritten(tmp_path, contents):
    cache_path = tmp_path / "invalid.pkl"
    cache_path.write_bytes(contents)

    with pytest.raises(ValueError, match="Cached shape|cached shape"):
        load_shape(
            cache_path,
            tmp_path,
            dataset="compat3d",
        )

    assert cache_path.read_bytes() == contents
