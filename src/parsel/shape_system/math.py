import itertools

import numpy as np
import sympy as sp

from .constants import (
    MOVE_LIMIT,
    NUMERIC_EVAL,
    QUANT_COMPARE_THRESHOLD_MODE_1,
    QUANT_COMPARE_THRESHOLD_MODE_2,
    TRIALS,
    UPPER_BOUND,
)


DIRECTIONS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

DIR_TO_NAME = {
    (1, 0, 0): "right",
    (-1, 0, 0): "left",
    (0, 1, 0): "up",
    (0, -1, 0): "down",
    (0, 0, 1): "front",
    (0, 0, -1): "back",
}


def _replace_numeric_symbols(expr, symbols, sample):
    return expr.xreplace(
        {
            symbol: sp.Float(sample[index])
            for index, symbol in enumerate(symbols)
        }
    )


def always_real(expr):
    symbols = list(expr.free_symbols)
    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT

    for trial_ind in range(TRIALS):
        expr_val = _replace_numeric_symbols(
            expr,
            symbols,
            values[trial_ind],
        )
        if not expr_val.is_real:
            return False
    return True


def always_within_bounds(expr):
    symbols = list(expr.free_symbols)
    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT

    for trial_ind in range(TRIALS):
        expr_val = _replace_numeric_symbols(
            expr,
            symbols,
            values[trial_ind],
        )
        if not sp.Abs(expr_val) < UPPER_BOUND:
            return False
    return True


def evaluate_equals_zero(delta_expr, order=1, mode=1, value=None):
    if mode == 1:
        threshold = QUANT_COMPARE_THRESHOLD_MODE_1
    elif mode == 2:
        threshold = QUANT_COMPARE_THRESHOLD_MODE_2
    elif mode == 3:
        threshold = value * QUANT_COMPARE_THRESHOLD_MODE_1
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")

    if order != 1:
        threshold = ((threshold**2) / order) ** (1 / 2)

    if isinstance(delta_expr, (sp.Float, sp.Integer)):
        return np.abs(float(delta_expr)) < threshold

    if not isinstance(delta_expr, sp.Expr):
        return np.abs(float(delta_expr)) < threshold

    if not NUMERIC_EVAL:
        return False

    symbols = list(delta_expr.free_symbols)
    if not symbols:
        raise ValueError("At least one expression must be symbolic.")

    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
    boundary_values = [0, MOVE_LIMIT]
    boundary_combinations = list(
        itertools.product(boundary_values, repeat=len(symbols))
    )
    values = np.vstack([values, boundary_combinations])

    for trial_ind in range(TRIALS):
        # Every replacement key is an exact free-symbol atom. ``xreplace``
        # therefore has the same meaning as ``subs`` here, while avoiding
        # SymPy's much more expensive algebraic substitution machinery.
        expr_val = _replace_numeric_symbols(
            delta_expr,
            symbols,
            values[trial_ind],
        )
        if not expr_val.is_real:
            return False
        if np.abs(float(expr_val)) > threshold:
            return False
    return True


def numeric_limited_eval(expr):
    symbols = list(expr.free_symbols)
    if not symbols:
        raise ValueError("At least one expression must be symbolic.")

    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
    return [
        _replace_numeric_symbols(expr, symbols, values[trial_ind])
        for trial_ind in range(TRIALS)
    ]


def rotation_matrix_sympy(axis, angle):
    """Return a 3D rotation matrix using Rodrigues' formula."""
    axis = axis / sp.sqrt(axis.dot(axis))
    u_x, u_y, u_z = axis
    cross_product = sp.Matrix(
        [
            [0, -u_z, u_y],
            [u_z, 0, -u_x],
            [-u_y, u_x, 0],
        ]
    )
    rotation = (
        sp.eye(3)
        + sp.sin(angle) * cross_product
        + (1 - sp.cos(angle)) * cross_product**2
    )
    rotation.simplify()
    return rotation


def assign_directions(point_set):
    """Map the faces, edges, and corners of a hexahedron to direction names."""
    faces = [
        (3, 2, 1, 0),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 4, 7, 3),
    ]
    direction_to_indices = {}
    directions = DIRECTIONS.copy()

    center = sp.ones(1, point_set.shape[0]) * point_set / point_set.shape[0]
    for face in faces:
        face_points = point_set[face, :]
        face_center = (
            sp.ones(1, face_points.shape[0])
            * face_points
            / face_points.shape[0]
        )
        normal = (face_center - center).normalized()
        stacked_normals = sp.Matrix.vstack(
            *[normal for _ in range(len(directions))]
        )
        stacked_directions = sp.Matrix.vstack(
            *[sp.Matrix(1, 3, direction) for direction in directions]
        )
        dot_products = stacked_normals.multiply_elementwise(stacked_directions)
        dot_products = dot_products * sp.ones(dot_products.shape[1], 1)
        max_ind = np.argmax(np.array(dot_products))
        max_direction = directions.pop(max_ind)
        direction_to_indices[DIR_TO_NAME[max_direction]] = face

    for first, second in itertools.combinations(direction_to_indices.keys(), 2):
        edge = tuple(
            sorted(
                set(direction_to_indices[first]).intersection(
                    direction_to_indices[second]
                )
            )
        )
        if len(edge) == 2:
            direction_to_indices[tuple(sorted([first, second]))] = edge

    face_names = [
        name for name in direction_to_indices.keys() if isinstance(name, str)
    ]
    for first, second, third in itertools.combinations(face_names, 3):
        corner = tuple(
            sorted(
                set(direction_to_indices[first]).intersection(
                    direction_to_indices[second],
                    direction_to_indices[third],
                )
            )
        )
        if len(corner) == 1:
            direction_to_indices[tuple(sorted([first, second, third]))] = corner

    return direction_to_indices
