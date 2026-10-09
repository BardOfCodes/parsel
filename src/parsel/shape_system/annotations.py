from string import Template
import sympy as sp
import numpy as np
from .constants import (RELATION_BROKEN, RELATION_REJECTED, RELATION_RETAINED, 
                        RELATION_STABLE, RELATION_UNCHECKED, 
                        PART_EDITED, PART_UNEDITED)
from .constants import (ANNOTATE_DIST_THRESHOLD, ANNOTATE_DP_THRESHOLD,
                        DIR_MAT, DIR_CODE, DIR_NAMES, N_DIR,
                        LOCAL_COORD_MAT, N_LOCAL_COORDS, COORD_LABELS, POINT_NAMES, PROMPT_SIGS)
## BASIC ANNOTATION

def round_expr(expr, num_digits=2):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})

def sympy_vec_to_str(vec):
    vec_entires = [x[0] for x in vec.tolist()]
    str_list = []
    for entry in vec_entires:
        if entry.is_real:
            str_list.append(str(f"{entry:.2f}"))
        else:
            str_list.append(str(entry))
    str_vec = "[" + ", ".join(str_list) + "]"
    return str_vec

def sympy_annotate_via_dp_vec(vec):
    # stack vec 6 times
    vec_s = sp.Matrix.vstack(
        *[
            vec,
        ]
        * N_DIR
    )
    # compute dot product
    dot_prod = vec_s.multiply_elementwise(DIR_MAT)
    dot_prod = [sum(dot_prod[i, :]) for i in range(N_DIR)]
    sel_ind = np.argmax(dot_prod)
    value = dot_prod[sel_ind]
    if value >= ANNOTATE_DP_THRESHOLD:
        annotation = DIR_NAMES[sel_ind]
    else:
        annotation = sympy_vec_to_str(vec)
    return annotation

def sympy_annotate_via_distance_vec(vec, scale=1.0):
    # stack vec 6 times
    vec_s = sp.Matrix.vstack(*[vec,]* N_LOCAL_COORDS)
    scaled_coord_mat = LOCAL_COORD_MAT * scale
    diff = vec_s - scaled_coord_mat.T
    dist = [diff[:, i].norm() for i in range(N_LOCAL_COORDS)]
    sel_ind = np.argmin(dist)
    value = dist[sel_ind]
    if value < ANNOTATE_DIST_THRESHOLD:
        annotation = COORD_LABELS[sel_ind]
    else:
        annotation = sympy_vec_to_str(vec)
    return annotation

# PART SPECIFIC ANNOTATION

def part_specific_annotate_via_distance_vec(vec, verts, part_label, use_prompt_sig=True):
    n_coords = verts.shape[1]
    vec_s = sp.Matrix.hstack(*[vec,]* n_coords)
    diff = vec_s - verts
    dist = [diff[:, i].norm() for i in range(n_coords)]
    sel_ind = np.argmin(dist)
    value = dist[sel_ind]
    if value < ANNOTATE_DIST_THRESHOLD:
        if use_prompt_sig:
            annotation = POINT_NAMES[sel_ind]
        else:
            annotation = Template(PROMPT_SIGS[sel_ind]).substitute({"part": part_label})
    else:
        annotation = sympy_vec_to_str(vec)
    return annotation

def part_specific_annotate_via_dp_vec(vec, use_prompt_sig=True):
    # stack vec 6 times
    vec_s = sp.Matrix.hstack(
        *[
            vec,
        ]
        * N_DIR
    )
    # compute dot product
    dot_prod = vec_s.multiply_elementwise(DIR_MAT.T)
    dot_prod = [sum(dot_prod[:, i]) for i in range(N_DIR)]
    sel_ind = np.argmax(dot_prod)
    value = dot_prod[sel_ind]
    if value >= ANNOTATE_DP_THRESHOLD:
        if use_prompt_sig:
            annotation = DIR_NAMES[sel_ind]
        else:
            annotation = DIR_CODE[sel_ind]
    else:
        annotation = sympy_vec_to_str(vec)
    return annotation

# PROMPT SPECIFIC ANNOTATION

init_shape_text_template = """# The input shape has the following sub-parts:
$part_list
# Use the variable `shape` to refer to the input shape.
# Use shape.get(label) to access part with label `label`.
"""
def initialize_text_description(part):
    # list of parts:
    part_list = part.partset
    part_list_str = (
        "shape.parts = set([\n    "
        + ",\n     ".join([str(x) for x in part_list])
        + "])"
    )
    # combine:
    substitute_dict = {"part_list": part_list_str}

    description = Template(init_shape_text_template).substitute(substitute_dict)
    return description

keep_fixed_template = """#### Sub-parts of the input shape

$part_list"""
def keep_fixed_shape_description(part, use_prompt_sig=True):
    # list of parts:
    part_list = part.partset
    if use_prompt_sig:
        part_list = [x.prompt_signature() for x in part_list]
    else:
        part_list = [x.signature() for x in part_list]
    part_list.sort()
    part_list_str = [
        f"{ind}. {part_name}" for ind, part_name in enumerate(part_list, 1)
    ]
    part_list_str = "\n".join(part_list_str)
    # list of relations:
    substitute_dict = {"part_list": part_list_str}

    description = Template(keep_fixed_template).substitute(substitute_dict)
    return description


shape_text_template_1 = """#### Sub-parts of the input shape

$part_list

#### Part Relations before editing

$relation_list"""
def shape_signature(part, use_prompt_sig=True):
    # list of parts:
    part_list = part.partset
    if use_prompt_sig:
        part_list = [x.prompt_signature() for x in part_list]
    else:
        part_list = [x.signature() for x in part_list]
    part_list.sort()
    part_list_str = [
        f"{ind}. {part_name}" for ind, part_name in enumerate(part_list, 1)
    ]
    part_list_str = "\n".join(part_list_str)
    # list of relations:
    relation_list = part.relations
    relation_list = [
        x
        for x in list(part.relations)
        if x.retention_state in [RELATION_UNCHECKED, RELATION_RETAINED]
    ]
    if use_prompt_sig:
        relation_list = [relation.prompt_signature() for relation in relation_list]
    else:
        relation_list = [relation.signature() for relation in relation_list]
    relation_list.sort()
    relation_list_str = [
        f"{ind}. {relation_name}" for ind, relation_name in enumerate(relation_list, 1)
    ]
    relation_list_str = "\n".join(relation_list_str)

    # combine:
    substitute_dict = {"part_list": part_list_str, "relation_list": relation_list_str}

    description = Template(shape_text_template_1).substitute(substitute_dict)
    return description


shape_text_template_2 = """The following parts may be edited in later steps:

$part_list"""
def shape_remaining_parts_str(shape, part):
    part_list = shape.partset
    part_list = [
        x.signature()
        for x in part_list
        if x.label != part.label and x.state == PART_UNEDITED
    ]
    if len(part_list) == 0:
        return ""
    else:
        part_list.sort()
        part_list_str = [
            f"{ind}. {part_name}" for ind, part_name in enumerate(part_list, 1)
        ]
        part_list_str = "\n".join(part_list_str)
        # combine:
        substitute_dict = {"part_list": part_list_str}

        description = Template(shape_text_template_2).substitute(substitute_dict)
        return description

shape_text_template_3 = """The following parts have not yet been edited:

$part_list"""
def shape_unedited_parts_str(shape):
    part_list = shape.partset
    part_list = [
        x.signature()
        for x in part_list
        if x.state == PART_UNEDITED
    ]
    part_list.sort()
    part_list_str = [
        f"{ind}. {part_name}" for ind, part_name in enumerate(part_list, 1)
    ]
    part_list_str = "\n".join(part_list_str)
    # combine:
    substitute_dict = {"part_list": part_list_str}

    description = Template(shape_text_template_3).substitute(substitute_dict)
    return description
