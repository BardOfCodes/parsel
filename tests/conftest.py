import os
import random

import numpy as np
import pytest
import sympy as sp

from parsel.shape_system.shape import Hexahedron, Part, PointFeature


# Several legacy equality checks use random numeric samples. Resetting both
# generators makes each characterization test repeatable.
os.environ.setdefault("PYTHONHASHSEED", "0")
random.seed(0)
np.random.seed(0)


UNIT_BOX_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


class TestOBB:
    """Minimal Open3D-OBB-compatible object for symbolic hierarchy tests."""

    def __init__(self, center, extent):
        self.center = np.asarray(center, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)

    def get_center(self):
        return self.center


@pytest.fixture(autouse=True)
def _reset_random_generators():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def box_factory():
    def make_box(
        label,
        *,
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        part_index=0,
    ):
        center = np.asarray(center, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        half_size = size / 2.0
        points = UNIT_BOX_CORNERS * half_size[None, :] + center[None, :]
        primitive = Hexahedron(points, obb=TestOBB(center, size))
        return Part(
            label,
            primitive,
            part_index=part_index,
            original_label=label,
        )

    return make_box


@pytest.fixture
def root_factory(box_factory):
    def make_root(children, *, label="shape", part_index=999):
        root = box_factory(
            label,
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            part_index=part_index,
        )
        root.sub_parts = set(children)
        for child in children:
            child.parent = root
        return root

    return make_root


@pytest.fixture
def point_feature_factory():
    def make_point_feature(part, weights):
        weights = sp.Matrix(1, 8, list(weights))
        return PointFeature(part.primitive, weights)

    return make_point_feature


@pytest.fixture
def evaluate_matrix():
    def evaluate(matrix, substitutions=None):
        if substitutions:
            matrix = matrix.subs(substitutions)
        return np.asarray(matrix, dtype=np.float64)

    return evaluate


@pytest.fixture
def assert_symbolic_matrix_equal():
    def assert_equal(actual, expected):
        actual = sp.Matrix(actual)
        expected = sp.Matrix(expected)
        assert actual.shape == expected.shape
        for value in actual - expected:
            assert sp.simplify(value) == 0

    return assert_equal
