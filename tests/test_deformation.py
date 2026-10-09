import numpy as np
import sympy as sp

from parsel.edit_dsl.edits import MAIN_VAR, PartTranslate
from parsel.shape_system.geometry import RightUnitVector


def test_cage_coordinates_transfer_proxy_translation_to_vertices(
    box_factory,
    evaluate_matrix,
):
    part = box_factory("box")
    edit = PartTranslate(part, direction=RightUnitVector(), amount=MAIN_VAR)
    edit.propagate()

    cage_coordinates = sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 8)] * 8,
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )
    static_vertices = cage_coordinates * part.primitive.static_expression()
    dynamic_vertices = cage_coordinates * part.primitive.dynamic_expression()

    static = evaluate_matrix(static_vertices)
    dynamic = evaluate_matrix(dynamic_vertices, {MAIN_VAR: 0.3})
    np.testing.assert_allclose(dynamic, static + np.array([0.3, 0, 0]))
