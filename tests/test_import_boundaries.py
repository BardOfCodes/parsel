import subprocess
import sys


def test_old_flat_module_paths_are_not_kept_as_aliases(tmp_path):
    script = """
import importlib.util

old_paths = [
    "parsel.dsl",
    "parsel.edits",
    "parsel.geometry",
    "parsel.io",
    "parsel.propagation",
    "parsel.relations",
    "parsel.shape",
]
remaining = [name for name in old_paths if importlib.util.find_spec(name)]
if remaining:
    raise SystemExit(f"old flat module paths still exist: {remaining}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_namespaced_core_and_propagation_do_not_import_optional_systems(
    tmp_path,
):
    script = """
import sys

import parsel
import parsel.data
import parsel.data.loading
import parsel.edit_dsl.language
import parsel.edit_propagation
import parsel.shape_system.shape

if hasattr(parsel, "Translate") or hasattr(parsel, "Part"):
    raise SystemExit("root package should not flatten subsystem APIs")

optional = {
    "anthropic",
    "mesh2sdf",
    "open3d",
    "openai",
    "torch",
    "trimesh",
}
loaded = sorted(optional.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional modules loaded by core import: {loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
