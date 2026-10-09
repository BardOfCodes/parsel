import sympy as sp
import numpy as np

from ..edit_dsl.edits import (
    MAIN_VAR,
    ChangeCount,
    ChangeDelta,
    PartRotate,
    PartTranslate,
)
from ..data.constants import ICP_THRESHOLD, MAX_ICP_ITER

from ..shape_system.math import always_real, always_within_bounds, evaluate_equals_zero
from ..shape_system.constants import MOVE_LIMIT
from ..shape_system.shape import Part
from .deformation import Deformer

DIRECTION_DOT_THRESHOLD = 0.98
COEFF_LIMIT = 100
DEFORM_THRESHOLD = 20

OPPOSITE_NAME = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "up": "down",
    "down": "up",
    "front": "back",
    "back": "front",
}

COSTS = {
    "ChangeCount": 10,
    "ChangeDelta": 11,
    "PartTranslate": 0,
    "PartRotate": 1,
    "StretchByFaceTranslate": 1.5,
    "PartScale1D": 2,
    "PartScale2D": 3,
    "PartScale3D": 4,
    "ShearByFaceTranslate": 5,
    "FaceScale1D": 6,
    "FaceScale2D": 6,
    "EdgeTranslate": 7,
    "EdgeScale1D": 8,
    "EdgeScale2D": 8,
    "FaceRotate": 8.5,
    "PartShear": 9,
}
INTERNAL_SYM_COST = {
    "ChangeCount": 10,
    "ChangeDelta": 11,
    "PartTranslate": 0,
    "PartRotate": 0,
    "StretchByFaceTranslate": 0,
    "PartScale1D": 0,
    "PartScale2D": 0,
    "PartScale3D": 0,
    "FaceScale1D": 1,
    "FaceScale2D": 1,
    "ShearByFaceTranslate": 2,
    "EdgeTranslate": 2,
    "EdgeScale1D": 2,
    "EdgeScale2D": 2,
    "FaceRotate": 2,
    "PartShear": 2,
}

RELATION_EDIT_ENERGY = {
    ChangeDelta: 0,
    ChangeCount: 1
}


def get_directions(part):
    og_directions = [
        sp.Matrix(1, 3, [1, 0, 0]),
        sp.Matrix(1, 3, [0, 1, 0]),
        sp.Matrix(1, 3, [0, 0, 1]),
    ]

    right_dir = (part.face_center('right') - part.center()).normalized()
    up_dir = (part.face_center('up') - part.center()).normalized()
    front_dir = (part.face_center('front') - part.center()).normalized()
    # Part Level Edits:
    directions = [right_dir, up_dir, front_dir]
    selected_dirs = []
    for cur_dir in og_directions:
        dots = [sp.Abs(cur_dir.dot(x)) for x in directions]
        if not np.max(dots) > DIRECTION_DOT_THRESHOLD:
            selected_dirs.append(cur_dir)
    directions = directions + selected_dirs
    return directions


def select_new_edits(part, old_edits_to_try, all_new_edits):
    selected_edits = []
    buffer = []
    for ind, edit_gen in enumerate(old_edits_to_try):
        redundant = False
        # new_edit = edit_class(operand=part, **edit_params)
        new_edit = edit_gen.employ(operand=part)
        for prev_edit in buffer:
            if new_edit == prev_edit:
                redundant = True
                break
        if not redundant:
            buffer.append(new_edit)

    for ind, edit_gen in enumerate(all_new_edits):
        redundant = False
        # new_edit = edit_class(operand=part, **edit_params)
        new_edit = edit_gen.employ(operand=part)
        for prev_edit in buffer:
            if new_edit == prev_edit:
                redundant = True
                break
        if not redundant:
            buffer.append(new_edit)
            selected_edits.append(edit_gen)
    return selected_edits


def merge_finalized_edits(part, all_new_edits, extra=None):
    selected_edits = []
    buffer = []
    add_extra = False
    if not extra is None:
        selected_extra = []
        add_extra = True

    for ind, edit_gen in enumerate(all_new_edits):
        redundant = False
        # new_edit = edit_class(operand=part, **edit_params)
        new_edit = edit_gen.employ(operand=part)
        for prev_edit in buffer:
            if prev_edit.merge_equal(new_edit):
                redundant = True
                break
        if not redundant:
            buffer.append(new_edit)
            selected_edits.append(edit_gen)
            if add_extra:
                selected_extra.append(extra[ind])
    if add_extra:
        return selected_edits, selected_extra
    else:
        return selected_edits

def sort_by_typed_distortion_energy(
    target_part,
    edit_candidates,
    extra=None,
    heuristic="intrinsic",
):

    add_extra = False
    if not extra is None:
        add_extra = True
        selected_extra = []
    # select top k, or reject more than 2x the energy of the best.
    distortion_energies = get_typed_distortion_energies(target_part, edit_candidates, heuristic=heuristic)
    indexed_distortion = {i: x for i, x in enumerate(distortion_energies)}
    sorted_indices = sorted(indexed_distortion, key=lambda x: indexed_distortion[x])
    selected_candidates = [edit_candidates[x] for x in sorted_indices]
    if add_extra:
        selected_extra = [extra[x] for x in sorted_indices]
        return selected_candidates, selected_extra
    
    return selected_candidates
    

def remove_flips(target_part, edit_candidates, extra=None, max_val=None):
    add_extra = False
    if not extra is None:
        add_extra = True
        selected_extra = []

    symbol = MAIN_VAR
    if max_val is None:
        max_val = MOVE_LIMIT
    selected_candidates = []
    indices = []
    primitive = target_part.primitive
    index_0 = primitive.name_to_indices[('back', 'down', 'left')]
    index_1 = primitive.name_to_indices[('back', 'left', 'up')]
    index_2 = primitive.name_to_indices[('back', 'down', 'right')]
    index_3 = primitive.name_to_indices[('down', 'front', 'left')]
    indices.append((index_0, index_1, index_2, index_3))

    direction_sets = []
    index_0, index_1, index_2, index_3 = indices[0]
    static_expr = primitive.static_expression()
    direction_1 = (static_expr[index_1, :] -
                   static_expr[index_0, :]).normalized()
    direction_2 = (static_expr[index_2, :] -
                   static_expr[index_0, :]).normalized()
    direction_3 = (static_expr[index_3, :] -
                   static_expr[index_0, :]).normalized()
    direction_sets.append((direction_1, direction_2, direction_3))

    for ind, candidate in enumerate(edit_candidates):
        no_flips = True
        edit = candidate.employ(operand=target_part)
        edit.propagate()

        dyn_expr = edit.operand.primitive.dynamic_expression()

        eval_expr = dyn_expr.subs({symbol: max_val})
        index_0, index_1, index_2, index_3 = indices[0]
        direction_1 = (eval_expr[index_1, :] -
                       eval_expr[index_0, :]).normalized()
        direction_2 = (eval_expr[index_2, :] -
                       eval_expr[index_0, :]).normalized()
        direction_3 = (eval_expr[index_3, :] -
                       eval_expr[index_0, :]).normalized()
        # previous
        prev_dir_1, prev_dir_2, prev_dir_3 = direction_sets[0]
        # check dot prod
        dot_1 = direction_1.dot(prev_dir_1)
        dot_2 = direction_2.dot(prev_dir_2)
        dot_3 = direction_3.dot(prev_dir_3)
        # Flip checker
        flip_checker_value = -0.9
        if dot_1 < flip_checker_value or dot_2 < flip_checker_value or dot_3 < flip_checker_value:
            no_flips = False
        if no_flips:
            selected_candidates.append(candidate)
            if add_extra:
                selected_extra.append(extra[ind])
        clean_up_motion(target_part, edit)

    if len(selected_candidates) == 0:
        # FALL BACK if all flips
        # Can happen when small objects flattenned.
        selected_candidates = edit_candidates
        if add_extra:
            selected_extra = extra

    if add_extra:
        # selected_extra = selected_extra[:15]
        return selected_candidates, selected_extra
    else:
        return selected_candidates

def remove_high_distortion(target_part, edit_candidates, extra=None, return_sorted=False):
    selected_edits = []
    add_extra = False
    if not extra is None:
        selected_extra = []
        add_extra = True
    # For reject consider with the ratio.

    static = target_part.primitive.static_expression()
    # A variant where this is driven by 
    static_np = np.asarray(static).astype(np.float32)
    v_map, F = get_vert_face_map(target_part)

    seq_points = static_np[v_map]
    #### HACK
    # index_0 = target_part.primitive.name_to_indices[('back', 'down', 'left')]
    # index_1 = target_part.primitive.name_to_indices[('back', 'left', 'up')]
    # index_2 = target_part.primitive.name_to_indices[('back', 'down', 'right')]
    # index_3 = target_part.primitive.name_to_indices[('down', 'front', 'left')]
    # vec_1 = static_np[index_1, :] - static_np[index_0, :]
    # vec_2 = static_np[index_2, :] - static_np[index_0, :]
    # vec_3 = static_np[index_3, :] - static_np[index_0, :]
    # all_vecs = np.stack([vec_1[0], vec_2[0], vec_3[0]], axis=0)
    # # norm
    # normed_vals = np.linalg.norm(all_vecs, axis=1)
    # scale = max(normed_vals)
    ###

    for ind, edit_option in enumerate(edit_candidates):
        edit = edit_option.employ(operand=target_part)
        edit.propagate()
        dynamic_expr = target_part.primitive.dynamic_expression()
        real_dynamic = dynamic_expr.subs({MAIN_VAR: MOVE_LIMIT})
        dynamic_np = np.asarray(real_dynamic).astype(np.float32)
        dynamic_points = dynamic_np[v_map]
        deform = Deformer(seq_points, dynamic_points, F)
        deform_energy = deform.energy # / scale

        if not deform_energy > DEFORM_THRESHOLD:
            selected_edits.append(edit_option)
            if add_extra:
                selected_extra.append(extra[ind])
        clean_up_motion(target_part, edit)

    if add_extra:
        return selected_edits, selected_extra
    else:
        return selected_edits


def remove_null_edits(all_new_edits, extra=None):
    selected_edits = []
    add_extra = False
    if not extra is None:
        selected_extra = []
        add_extra = True
    for ind, edit_gen in enumerate(all_new_edits):
        amount = edit_gen.amount
        if not evaluate_equals_zero(amount):
            selected_edits.append(edit_gen)
            if add_extra:
                selected_extra.append(extra[ind])
    if len(selected_edits) == 0:
        # Fall back when all options are nulls
        selected_edits = all_new_edits
        if add_extra:
            selected_extra = extra
    if add_extra:
        return selected_edits, selected_extra
    else:
        return selected_edits


def valid_solution(solution):
    # REAL + Withing Bounds + COEFF LIMIT
    realness = always_real(solution)
    valid = False
    if realness:
        within_bounds = always_within_bounds(solution)
        if within_bounds:
            # check coefficient of X - the abs value should be less than LIMIT.
            coef = solution.coeff(MAIN_VAR)
            if sp.Abs(coef) < COEFF_LIMIT:
                valid = True
    return valid


def clean_up_motion(target_part, edit_):
    if isinstance(target_part, Part):
        target_part.primitive.edit_sequence.remove(edit_)
    else:
        target_part.edit_sequence.remove(edit_)

def _amount_based_distortion_energies(part_to_edit, edit_options):
    all_energies = []
    cost = [COSTS[x.employ(part_to_edit).edit_type] for x in edit_options]
    if cost:
        min_cost = min(cost)
    else:
        min_cost = 0
    cost = [x if x == min_cost else np.inf for x in cost]
    amount = [sp.Abs(i.amount) for i in edit_options]
    amount_eval = [x.subs({MAIN_VAR: 0.3}) for x in amount]
    cost = [cost[i] + amount_eval[i] for i in range(len(cost))]
    # cost = amount_eval
    return cost

def _intrinsic_distortion_energies(part_to_edit, edit_options):

    all_energies = []
    cost = [INTERNAL_SYM_COST[x.employ(part_to_edit).edit_type] for x in edit_options]
    # cost = [0 for x in edit_options]
    if cost:
        min_cost = min(cost)
    else:
        min_cost = 0
    cost = [x if x == min_cost else np.inf for x in cost]

    valid_edit_inds = [i for i, x in enumerate(cost) if x < np.inf]

    # Measure the static volume

    static = part_to_edit.primitive.static_expression()
    # A variant where this is driven by 
    static_np = np.asarray(static).astype(np.float32)
    v_map, F = get_vert_face_map(part_to_edit)
    seq_points = static_np[v_map]

    # Prefer Translation over rotation
    for ind in valid_edit_inds:
        if issubclass(edit_options[ind].edit_class, PartTranslate):
            cost[ind] += 0
        elif issubclass(edit_options[ind].edit_class, PartRotate):
            cost[ind] += 1
        else:
            cost[ind] += 2
    # what to do when there are multiple "translates"?
    # Ideally there shouldn't be multiple with similar energy but different results.,


    for ind in valid_edit_inds:
        edit_option = edit_options[ind]
        edit = edit_option.employ(operand=part_to_edit)
        edit.propagate()
        dynamic_expr = part_to_edit.primitive.dynamic_expression()
        real_dynamic = dynamic_expr.subs({MAIN_VAR: MOVE_LIMIT})
        dynamic_np = np.asarray(real_dynamic).astype(np.float32)
        dynamic_points = dynamic_np[v_map]
        deform = Deformer(seq_points, dynamic_points, F)
        cost[ind] += deform.energy
        clean_up_motion(part_to_edit, edit)
    return cost


def get_volume(points, tet_seqs):
    volume = 0
    for i in range(6):
        cur_points = points[tet_seqs[i], :]
        v1 = cur_points[0, :] - cur_points[3, :]
        v2 = cur_points[1, :] - cur_points[3, :]
        v3 = cur_points[2, :] - cur_points[3, :]
        cur_volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6
        volume += cur_volume
    return volume

def get_tet_seqs(part_to_edit):

    pointer_dict = part_to_edit.primitive.name_to_indices

    # use sorted order
    pt_0_0_0 = pointer_dict[tuple(sorted(('back', 'down', 'left')))][0]
    pt_1_0_0 = pointer_dict[tuple(sorted(('back', 'down', 'right')))][0]
    pt_0_1_0 = pointer_dict[tuple(sorted(('back', 'up', 'left')))][0]
    pt_0_0_1 = pointer_dict[tuple(sorted(('front', 'down', 'left')))][0]
    pt_1_1_0 = pointer_dict[tuple(sorted(('back', 'up', 'right')))][0]
    pt_1_0_1 = pointer_dict[tuple(sorted(('front', 'down', 'right')))][0]
    pt_0_1_1 = pointer_dict[tuple(sorted(('front', 'up', 'left')))][0]
    pt_1_1_1 = pointer_dict[tuple(sorted(('front', 'up', 'right')))][0]

    tet_seqs = [
        [pt_0_0_0, pt_0_0_1, pt_0_1_1, pt_1_1_1],
        [pt_0_0_0, pt_0_1_0, pt_0_1_1, pt_1_1_1],
        [pt_0_0_0, pt_0_0_1, pt_1_0_1, pt_1_1_1],
        [pt_0_0_0, pt_1_0_0, pt_1_0_1, pt_1_1_1],
        [pt_0_0_0, pt_0_1_0, pt_1_1_0, pt_1_1_1],
        [pt_0_0_0, pt_1_0_0, pt_1_1_0, pt_1_1_1],
    ]
    tet_seqs = np.array(tet_seqs)
    return tet_seqs

def get_vert_face_map(part_to_edit):


    pointer_dict = part_to_edit.primitive.name_to_indices

    # use sorted order
    pt_0_0_0 = pointer_dict[tuple(sorted(('back', 'down', 'left')))][0]
    pt_1_0_0 = pointer_dict[tuple(sorted(('back', 'down', 'right')))][0]
    pt_0_1_0 = pointer_dict[tuple(sorted(('back', 'up', 'left')))][0]
    pt_0_0_1 = pointer_dict[tuple(sorted(('front', 'down', 'left')))][0]
    pt_1_1_0 = pointer_dict[tuple(sorted(('back', 'up', 'right')))][0]
    pt_1_0_1 = pointer_dict[tuple(sorted(('front', 'down', 'right')))][0]
    pt_0_1_1 = pointer_dict[tuple(sorted(('front', 'up', 'left')))][0]
    pt_1_1_1 = pointer_dict[tuple(sorted(('front', 'up', 'right')))][0]
    # V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    vert_seq = np.array([pt_0_0_0, pt_1_0_0, pt_1_1_0, pt_0_1_0, pt_0_0_1, pt_1_0_1, pt_1_1_1, pt_0_1_1])
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2], [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3]])
    return vert_seq, F

def get_best_edit_ind(part_to_edit, ed_options):
    cost = [COSTS[x.employ(part_to_edit).edit_type] for x in ed_options]
    min_cost = min(cost)
    # print(cost)
    min_variants = [i for i, x in enumerate(cost) if x == min_cost]
    if len(min_variants) == 1:
        min_ind = min_variants[0]
    else:
        # based on the amount of edit. select less edit
        amount = [sp.Abs(ed_options[i].amount) for i in min_variants]
        amount_eval = [x.subs({MAIN_VAR: MOVE_LIMIT}) for x in amount]
        # sel_options = [ed_options[i] for i in min_variants]
        # amount_eval = get_typed_distortion_energies(part_to_edit, sel_options)
        min_ind = min_variants[amount_eval.index(min(amount_eval))]

    return min_ind

def get_typed_distortion_energies(
    part_to_edit,
    edit_options,
    heuristic="intrinsic",
):
    aliases = {
        "ARAP": "intrinsic",
        "ARAP_INTRINSIC_SYM": "intrinsic",
        "ARAP_ONLY": "amount",
    }
    heuristic = aliases.get(heuristic, heuristic)
    if heuristic == "intrinsic":
        return _intrinsic_distortion_energies(
            part_to_edit,
            edit_options,
        )
    if heuristic == "amount":
        return _amount_based_distortion_energies(
            part_to_edit,
            edit_options,
        )
    raise ValueError("heuristic must be 'intrinsic' or 'amount'")
