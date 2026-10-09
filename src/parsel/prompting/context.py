import sympy as sp
from ..shape_system.relations import ReflectionSymmetry, TranslationSymmetry, RotationSymmetry, PrimitiveRelation, FeatureRelation
from ..shape_system.constants import (
    PART_ACTIVE,
    PART_INACTIVE,
    PART_EDITED,
    PART_UNEDITED,
    RELATION_REJECTED,
    RELATION_RETAINED,
    RELATION_UNCHECKED,
)
from ..edit_dsl.edits import (DummyReflectionUpdateEdit, KeepFixed, DummyContactParamUpdate, get_direction_string, get_bidirection_string)
PART_LIMIT = 5
from collections import defaultdict

class DotBot:
    def __init__(self):
        self.label = "..."
        self.sub_parts = set()


def shape_to_file_hier(part, indent_level=0, with_bbox=False):
    """
    Recursively builds a string representation of the part hierarchy.

    Args:
    part (Part): The part to be represented.
    indent_level (int): The current level of indentation for sub-parts.

    Returns:
    str: A string representation of the part hierarchy.
    """
    if part.label == "ground":
        return ''
    # Base indent
    indent = '    ' * indent_level

    # Start with the label of the current part
    result = f"{indent}{part.label}\n"

    # If there are sub-parts, process each of them
    if part.sub_parts:
        # Add a slash to indicate a directory-like structure
        if with_bbox:
            if part.bbox is not None:
                result = f"{indent}{part.label} (bbox: {part.bbox})"
            else:
                result = f"{indent}{part.label}"
        else:
            result = f"{indent}{part.label}"

        # Recursively call function for each sub-part
        # check if the part has
        if hasattr(part, "core_relation"):
            if part.core_relation.state[2] != RELATION_REJECTED:
                if isinstance(part.core_relation, TranslationSymmetry):
                    result += " (translation symmetry)/\n"
                elif isinstance(part.core_relation, RotationSymmetry):
                    result += " (rotation symmetry)/\n"
            sub_parts = [x.part for x in part.core_relation.primitives]
            if len(sub_parts) > PART_LIMIT:
                sub_parts = sub_parts[:3] + [DotBot()] + sub_parts[-1:]
            sort = False
        else:
            result += "/\n"
            sub_parts = [x for x in part.sub_parts]
            # Need to mush together parts with same labels
            sub_parts_dict = defaultdict(list)
            for sub_part in sub_parts:
                mini_tag = "_".join(sub_part.label.split("_")[:-1])
                sub_parts_dict[mini_tag].append(sub_part)
            sort = True
            new_sub_parts = []
            for key, item_list in sub_parts_dict.items():
                item_list.sort(key=lambda x: x.label)
                if len(item_list) > PART_LIMIT:
                    new_sub_parts.extend(item_list[:3] + [DotBot()] + item_list[-1:])
                    sort = False
                else:
                    new_sub_parts.extend(item_list)
            sub_parts = new_sub_parts
        result_set = []
        for sub_part in sub_parts:
            result_set.append(shape_to_file_hier(sub_part, indent_level + 1))
        if sort:
            result_set.sort()
        result += ''.join(result_set)
    if indent_level == 0:
        result = f"```markdown\n{result}```\n"
    return result

def get_relation_in_detail(relation, shape, add_secondary=False, edit_code=False):

    relation_definition = relation.signature()
    relation_definition = f"The relation under consideration is: {relation_definition}."
    # Get all relation betweeen parts of this relation and others too.
    if add_secondary:
        relevant_parts = relation.parts
        part_level_info = []
        for part in relevant_parts:
            relations = part.all_relations(only_active=True)
            retained_relations = [x for x in relations if x.state[2] == RELATION_RETAINED and x != relation]
            unchecked_relations = [x for x in relations if x.state[2] == RELATION_UNCHECKED and x != relation]
            part_info = ""
            if retained_relations:
                part_info += f"\nThe {part.full_label} has the following additional relations:\n"
                for ind, rel in enumerate(retained_relations):
                    part_info += f"\n{ind+1}. {rel.signature()}"    
            if unchecked_relations:
                if retained_relations:
                    part_info += "\n"
                part_info += f"\nThe {part.full_label} has the following unchecked relations:\n"
                for ind, rel in enumerate(unchecked_relations):
                    part_info += f"\n{ind+1}. {rel.signature()}"
            part_level_info.append(part_info)
        part_level_info = "\n".join(part_level_info)
    else:
        part_level_info = ""

    # Get all distance 1 relations -> 
    if not isinstance(relation, PrimitiveRelation):
        part = relation.get_unedited_parts()
        edited_part = [x for x in relation.parts if x not in part][0]
        edit = edited_part.primitive.edit_sequence[0]
        if edit_code:
            edit_signature = edit.code_signature(shape)
        else:
            edit_signature = edit.signature(shape)
        edit_on_parts = f"The part {edited_part.full_label} involved in this relation is being edited as follows: {edit_signature}"
    else:
        part = relation.get_unedited_parts()
        edited_part = [x for x in relation.parts if x not in part]
        if len(edited_part) > 0:
            edited_part = edited_part[0]
            edit = edited_part.primitive.edit_sequence[0]
            if edit_code:
                edit_signature = edit.code_signature(shape)
            else:
                edit_signature = edit.signature(shape)
            edit_on_parts = f"\nThe part {edited_part.full_label} involved in the relation is being edited as follows: {edit_signature}"
        else:
            edit_on_parts = f"None of the parts under in the {relation.parent_part.full_label} have an existing edit."

    parts_to_join = [relation_definition]
    if len(edit_on_parts) > 0:
        parts_to_join.append(edit_on_parts)
    if len(part_level_info) > 0:
        parts_to_join.append(part_level_info)

    relation_in_detail = "\n".join(parts_to_join)
    return relation_in_detail


def get_part_in_details(part_to_edit, shape, edit_code=False):

    part_definition = f"The part under consideration is: {part_to_edit.full_label}."
    # Get all the relations of the part:
    relations = part_to_edit.all_relations(only_active=True)
    retained_relations = [x for x in relations if x.state[2] == RELATION_RETAINED]
    unchecked_relations = [x for x in relations if x.state[2] == RELATION_UNCHECKED]
    part_info = ""
    other_parts = []
    if retained_relations:
        part_info += f"\nThe {part_to_edit.full_label} has the following enforced relations:\n"
        for ind, rel in enumerate(retained_relations):
            if isinstance(rel, PrimitiveRelation):
                rel_str = f"\n{ind+1}. {rel.signature()}"
            else:
                points = rel.features[0].static_expression()
                direction = sp.ones(1, points.shape[0]) * points / points.shape[0] - part_to_edit.primitive.center()
                direction_str = get_direction_string(direction, part_to_edit.primitive)
                # 
                rel_str = f"\n{ind+1}. {rel.signature()} on {part_to_edit.label}'s {direction_str} direction."
            part_info += rel_str
            other_parts.extend([x for x in rel.parts if x != part_to_edit])
        part_info += "\n"

    if unchecked_relations:
        part_info += f"\nThe {part_to_edit.full_label} has the following unchecked relations:\n"
        for ind, rel in enumerate(unchecked_relations):
            if isinstance(rel, PrimitiveRelation):
                rel_str = f"\n{ind+1}. {rel.signature()}"
            else:
                rel_str = ""
            part_info += rel_str
            other_parts.extend([x for x in rel.parts if x != part_to_edit])
        part_info += "\n"
    
    # Get all the 1 distance edits from the part:
    edit_information = f"The following editing constraints are employed on parts with a relation to {part_to_edit.full_label}:\n"
    count = 0
    for ind, part in enumerate(other_parts):
        if len(part.primitive.edit_sequence) > 0:
            for edit in part.primitive.edit_sequence:
                if edit_code:
                    edit_signature = edit.code_signature(shape)
                else:
                    edit_signature = edit.signature(shape)
                edit_information += f"\n{count+1}. {edit_signature}"
                count += 1
    
    part_in_detail = "\n".join([part_definition, part_info, edit_information])
    
    return part_in_detail


def get_unedited_parts(shape):
    part_list = []
    for part in shape.partset:
        if part.state[0] == PART_ACTIVE and part.state[1] == PART_UNEDITED:
            part_list.append(part)
    if len(part_list) == 0:
        unedit_string = ""
    elif len(part_list) > 10:
        unedit_string = ""
    else:
        unedit_string = "The following parts may be edited in future steps:\n"
        for ind, part in enumerate(part_list):
            unedit_string += f"\n{ind+1}. {part.full_label}"
        
    return unedit_string

def get_all_edits_in_detail(all_edits, shape):
    edit_list = []
    count = 0
    for ind, edit in enumerate(all_edits):
        if not isinstance(edit, (KeepFixed, DummyReflectionUpdateEdit, DummyContactParamUpdate)):
            edit_list.append(f"{count+1}. {edit.signature(shape)}")
            count += 1
    edit_string = "The following part-level edits will be performed to fulfill the edit request:\n\n"
    edit_string += "\n".join(edit_list)
    return edit_string

def get_options_in_detail(part_to_edit, edit_options, shape):
    # Basic Version:
    option_list = []
    for ind, option in enumerate(edit_options):
        temp_edit = option.employ(operand=part_to_edit)
        option_list.append(f"{ind+1}. {(temp_edit.__class__.__name__)}: {temp_edit.signature(shape)}")
    
    options_string = f"The edit candidates for {part_to_edit.full_label} are listed below:\n\n"
    options_string += "\n".join(option_list)
    return options_string

def generate_least_breaking_str(least_breaking, shape):
    edit, broken_relations = least_breaking[0]
    edit_string = edit.signature(shape)

    broken_relations_list = []
    for ind, rel in enumerate(broken_relations):
        br_rel_str = f"\n{ind+1}. {rel.signature()}"
        broken_relations_list.append(br_rel_str)

    broken_relations_str = "\n".join(broken_relations_list)

    info_1 = f"The following minimally relation breaking edit is available:"
    info_2 = "This edit breaks the following relation(s):"
    chunks = [info_1, edit_string, info_2, broken_relations_str]
    least_breaking_str = "\n".join(chunks)
    return least_breaking_str



def get_relation_in_detail_init(relation, shape, add_secondary=False, edit_code=False):

    relation_definition = relation.signature()
    relation_definition = f"The relation under consideration is: {relation_definition}."
    # Get all relation betweeen parts of this relation and others too.
    if add_secondary:
        relevant_parts = relation.parts
        part_level_info = []
        for part in relevant_parts:
            relations = part.all_relations(only_active=True)
            retained_relations = [x for x in relations if x.state[2] == RELATION_RETAINED and x != relation]
            unchecked_relations = [x for x in relations if x.state[2] == RELATION_UNCHECKED and x != relation]  
            if unchecked_relations:
                if retained_relations:
                    part_info += "\n"
                part_info += f"\nThe {part.full_label} has the following relations that may be enforced:\n"
                for ind, rel in enumerate(unchecked_relations):
                    part_info += f"\n{ind+1}. {rel.signature()}"
            part_level_info.append(part_info)
        part_level_info = "\n".join(part_level_info)
    else:
        part_level_info = ""
    
    # Specify the direction of the relation
    if isinstance(relation, ReflectionSymmetry):
        bidir_string = get_bidirection_string(relation.plane.normal, shape.primitive)
        plane_str = f"The reflection plane's normal is in the {bidir_string} direction."
        relation_definition = relation_definition + f" {plane_str}"
    elif isinstance(relation, TranslationSymmetry):
    # Get all distance 1 relations -> 
        bidir_string = get_bidirection_string(relation.delta, shape.primitive)
        direction_str = f"The translation symmetry is along the {bidir_string} direction."
        relation_definition = relation_definition + f" {direction_str}"
    elif isinstance(relation, RotationSymmetry):
        bidir_string = get_bidirection_string(relation.axis.direction, shape.primitive)
        axis_str = f"The rotation symmetry axis is along the {bidir_string} axis."
        relation_definition = relation_definition + f" {axis_str}"

    return relation_definition
