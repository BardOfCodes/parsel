from .settings import NOCOT, WITH_EXAMPLES
rot_trans_base_overview = """# Overview

You are a 3D shape editing assistant. You are given a 3D object composed of labeled parts with relations between them. The shape is being edited according to a user's edit request. You must specify if the given symmetry relation will be broken or maintained as shape is edited.

## Output Specification

You must specify the `relation_state` variable as a integer (should be a one of the following options). The variable must be specified in a code snippet in the following format:

```python
relation_state = 0 # or 1
# where 
# 0 implies the relation will be broken, 
# 1 implies the relation will be maintained
```
"""
ref_base_overview = """# Overview

You are a 3D shape editing assistant. You are given a 3D object composed of labeled parts with relations between them. The shape is being edited according to a user's edit request. You must specify if the given symmetry relation will be kept-as-is, or broken/updated as the shape is edited.

## Output Specification

You must specify the `relation_state` variable as a integer (should be a one of the following options). The variable must be specified in a code snippet in the following format:

```python
relation_state = 0 # or 1
# where 
# 0 implies the relation will be broken/updated, 
# 1 implies the relation will be kept as is
```
"""

specifications = """## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification

## User's Edit Request

$edit_request

## Relation under Consideration

$relation_in_detail
"""

reflection_steps = """## Steps for Evaluating Relationships in Edits

1. **Interpreting the Edit Request**: Analyze the edit request to understand the intended modifications to the related parts.

2. **Evaluating Dissimilar Edits**: If one part remains unchanged while another is edited, or if the parts are edited differently, the relationship is broken. In these cases, set `relation_state` to 0.

3. **Assessing Similar Edits**: If both parts are edited similarly, the relationship might still be intact. Proceed to the next step for further evaluation.

4. **Considering the Reflection Plane**: Assess whether the edits are symmetrically aligned with the reflection plane. When the edits are symmetric about the reflection plane, the relationship is maintained. You can set `relation_state` to 1 in this case.

5. **Consider Translation Carefully**: If the parts are moved along the direction of the reflection plane's normal, the relationship will *always* need to be updated. You *must* set `relation_state` to 0 in this case. When the parts move in opposite directions or move perpendicular to the plane's normal, the relationship remains intact (`relation_state` = 1).
"""

rot_trans_steps = """## Steps for Relation Evaluation

1. **Interpreting the Edit Request**: Carefully analyze the edit request to understand how the parts involved in the relation will be modified to fulfill the user's instructions.

2. **Evaluating Dissimilar Edits**: If some parts remain unchanged while others are edited, or if all the parts are edited differently, the relationship is broken. In these cases, set `relation_state` to 0.

3. **Assessing Similar Edits**: If all parts are edited in a similar manner, of if the count of the parts is changed, the relationship is maintained, with updated parameterization. In this case, set `relation_state` to 1.
"""

guidelines_list = [
    "**Analytical Justification**: Justify your decision in detail, considering the edit request.",
    "**Code Format**: Your decision should be presented as a single Python code snippet setting the `relation_state` variable as an Integer.",
    "**Avoid Breaking Relations**: In general, relations are broken only when it is impossible to fulfill the edit request without breaking the relation. For instance, when changing number of sub-parts in a relation, the parts are removed/inserted such that the symmetry is maintained.",
]

end = """Now, please explain your reasoning step-by-step and specify the `relation_state` variable by following the instructions above.
"""

ref_examples = """## Examples

**Example 1**

Edit Request: "I want to bring the legs of the table closer together."

Relation Under Consideration: Reflection symmetry between the `legs/leg_front_right` and `legs/leg_front_left` with the reflection plane normal along the left-right direction.

Solution: Both the legs are moving closer together along the left-right therefore its not broken. Since they are moving in opposite directions along the reflection plane's normal, the relation will be kept as it is. Therefore, 

```python
relation_state = 1
```

**Example 2**

Edit Request: "I want move the front leg foward while keeping the back leg as it is."

Relation Under Consideration: Reflection symmetry between the `legs/leg_back_right` and `legs/leg_front_right` with the reflection plane normal along the front-back direction.

Solution: The legs are moving in dissimilar ways, the relation will be broken. Therefore,

```python
relation_state = 0
```

**Example 3**

Edit Request: "I want to shift the back slats further up.

Relation Under Consideration: Reflection symmetry between the `slats/slat_left` and `slats/slats_right` with the reflection plane normal along the right-left direction.

Solution: The slats need to be moved in a similar fasion. Since they are being moved perpendicular to the reflection plane's normal, the relation will be kept as it is. Therefore,

```python
relation_state = 1
```

**Example 4**

Edit Request: "I want shift the bars between the legs further up."

Relation Under Consideration: Reflection symmetry between the `bars/bar_up` and `bars/bar_down` with the reflection plane normal along the up-down direction.

Solution: The bars need to be moved in a similar fasion. Since they are being moved in the same direction as the reflection plane's normal, the parts will no longer be symmetric about the original reflection plane. If the edit was symmetric, then one part would have to move up and one down, which would conflict with the user's edit request. Therefore,

```python
relation_state = 0
```

**Example 5**

Edit Request: "I want lower the top panel to contract the shelf."

Relation Under Consideration: Reflection symmetry between the "shelf_0" and "shelf_1" with the reflection plane normal along the up-down direction.

Solution: shelf_0" and "shelf_1" will be moved down as the shelf is contracted. Though they may remain symmetric, the reflection plane will also need to be moved down with them. Therefore, the parts will no longer be symmetric about the original reflection plane. Therefore,

```python
relation_state = 0
```
"""

rot_trans_examples = """## Examples

**Example 1**

Edit Request: "I want to rotate the legs of the table."

Relation Under Consideration: Rotational symmetry between the sub-parts of `legs` with the axis of rotation along the up-down direction.

Solution: The legs will be rotated in a similar manner, therefore the symmetry will be maintained. Hence,

```python
relation_state = 1
```

**Example 2**

Edit Request: "I want to make the central slat longer while keeping the side slats as they are."

Relation Under Consideration: Translational symmetry between the `slats/slat_left`, `slats/slat_center`, `slats/slat_left` with the direction of translation along the left-right direction.

Solution: The slats will scale in dissimilar ways, hence the relation will be broken. Therefore,

```python
relation_state = 0
```

**Example 3**

Edit Request: "I want to increase the distance between the legs of the swivel chair."

Relation Under Consideration: Rotational symmetry between the `legs/leg_back_right`, `legs/leg_back_left`, `legs/leg_front_right`, `legs/leg_front_left` with the axis of rotation along the up-down direction.

Solution: Since the distance bewteen the legs will be changed in a symmetric fashion, the relation will be maintained with an updated parameterization. Therefore,

```python
relation_state = 1
```

**Example 4**

Edit Request: "I want to increase the number of in the back."

Relation Under Consideration: Translational symmetry between the `slats/slat_left`, `slats/slat_center`, `slats/slat_right` with the direction of translation along the up-down direction.

Solution: As the number of slats are being changed, the relation is maintained, although with updated parameterization. Therefore,

```python
relation_state = 1
```
"""

def ref_instructions(nocot=NOCOT, with_examples=WITH_EXAMPLES):
    if nocot:
        selected_guidelines = guidelines_list[1:]
    else:
        selected_guidelines = guidelines_list
    guideline_str = "## Guidelines\n\n"
    if with_examples:
        start = [ref_base_overview, ref_examples, specifications, reflection_steps]
    else:
        start = [ref_base_overview, specifications, reflection_steps]
    start = "\n".join(start)
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([start, guideline_str, end])
    return instructions


def rot_trans_instructions(nocot=NOCOT, with_examples=WITH_EXAMPLES):
    if nocot:
        selected_guidelines = guidelines_list[1:]
    else:
        selected_guidelines = guidelines_list
    guideline_str = "## Guidelines\n\n"
    if with_examples:
        start = [rot_trans_base_overview, rot_trans_examples, specifications, rot_trans_steps]
    else:
        start = [rot_trans_base_overview, specifications, rot_trans_steps]
    start = "\n".join(start)
    for ind, guideline in enumerate(selected_guidelines):
        if ind == len(selected_guidelines) - 1:
            guideline_str += f"{ind + 1}. {guideline}"
        else:
            guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([start, guideline_str, end])
    return instructions