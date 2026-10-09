from .settings import NOCOT, WITH_EXAMPLES
start_instructions = """# Overview

You are a 3D shape editing AI assistant. You are given a 3D $shape_class composed of labeled parts, and a user's edit request in natural language. Your task is to translate the user's edit request into a edit-solver configuration. In this step, you must specify the part-level type-hints provided by the user.

## Output Specification

Specify a dictionary `type_hints` in python which specifies a mapping from a part's full name to the edit type hint provided in the edit request. The dictionary should be specified in a python code snippet in the following format:

```python
type_hints = {
    "part_full_name" : "type_hint" # must be one of the type_hint_options
    ...
}
```

 The type hints must be one of the `type_hint_options` shown below:

```python
type_hint_options = ['keep_fixed', 'translate', 'tilt', 'scale', 'change_count', 'change_delta']

```

`change_count` and `change_delta` are to be used only for parts with translation or rotation symmetry. When editing the object, `change_count` specifies that the number of instances in a symmetry group must be changed, and `change_delta` specifies that the distance between the instances should be changed. When the user wants to keep the count of instances in a symmetry group the same, the type hint should be `change_delta`."""

examples = """## Examples

Note that part names must be dependant on the shape specification provided in the prompt. In the following examples we only provide the edit request and the expected type hints.

Edit Request: "I want to move the front legs backwards."

Type Hints:

```python
type_hints = {
    "legs/leg_front_left" : "translate",
    "legs/leg_front_left" : "translate",
}
```

Edit Request: I want to make the tabletop wider, while keeping the table connectors as they are.

Type Hints:

```python
type_hints = {
    "tabletop": "scale",
    "table_connector_left": "keep_fixed",
    "table_connector_right": "keep_fixed",
    "table_connector_front": "keep_fixed",
    "table_connector_back": "keep_fixed",
}
```

Edit Request: I want to move the lower mattress upwards while keeping the top mattress as it is.

Type Hints:

```python
type_hints = {
    "mattress_lower": "translate",
    "mattress_upper": "keep_fixed",
}
```

Edit Request: I want to make the sofa back and seat wider while keeping the cushions as they are.

Type Hints:

```python
type_hints = {
    "sofa_back": "scale",
    "sofa_seat": "scale",
    "cushion_left": "keep_fixed",
    "cushion_right": "keep_fixed",
    "cushion_back": "keep_fixed",
}
```

Edit Request: I want to rotate the back table legs inwards while keeping the front table legs as they are.

Type Hints:

```python
type_hints = {
    "legs/leg_back_left": "tilt",
    "legs/leg_back_right": "tilt",
    "legs/leg_front_left": "keep_fixed",
    "legs/leg_front_right": "keep_fixed",
}
```

Edit Request: I want to increase the number of legs for this bed without changing the bed boards.

Type Hints:

```python
type_hints = {
    "legs": "change_count",
    "head_board": "keep_fixed",
    "side_boards/side_board_left": "keep_fixed",
    "side_boards/side_board_right": "keep_fixed",
    "foot_board": "keep_fixed",
}
```

Edit Request: I want to lengthen the back without changing any other parts.

Type Hints:

```python
type_hints = {
    "back": "scale",
    "seat": "keep_fixed",
    "armrest_left": "keep_fixed",
    "armrest_right": "keep_fixed",
    "legs": "keep_fixed",
}
```

Edit Request: I want to contract the top panel of the shelfs in the left-right direction, while retaining the number of drawers and handles.

Type Hints:

```python
type_hints = {
    "top_panel": "scale",
    "drawers": "change_delta",
    "handles": "change_delta",
}
```

Edit Request: I want to move the left-part of the armrest further outwards while keeping the right arm-rest as it is.

Type Hints:

```python
type_hints = {
    "armrest_left": "scale",
    "armrest_right": "keep_fixed",
}
```

Edit Request: I want to shorten the legs by moving their tops down.

Type Hints:

```python
type_hints = {
    'leg_front_right': 'scale',
    'leg_front_left': 'scale',
    'leg_back_right': 'scale',
    'leg_back_left': 'scale',
}
```"""

shape_specification = """Please specify the `type_hints` for the following object according to the user's edit request.

## Shape Specification

$shape_specification
## User's Edit Request

$edit_request

## Steps for Identifying the type hints

1. **Identify Explicitly stated parts**: Determine what parts are *explicitly* referred to in the edit request. Only these parts should be considered for the dictionary.

2. **Identify the requested edit type**: For the *explicitly* referred parts, analyse the edit request to determine the type-hint provided by the user.

3. **Add to the dictionary**: Now add these parts with their type-hint to the dictionary.

## Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explain your solution along the way.

2. **Code Format**: Specify the dictionary in a single Python code snippet, strictly adhering to the instructions above.

3. **Face/Edge Movement**: If the edit request specifies movement of a subpart of a part, the type hint should be 'scale'.

4. **Part Full Names**: A part's full name is its path from the root of the shape. For e.g., if the `back_slat_left` is under `back_slats`, then full name is "back_slats/back_slat_left". 

5. **Using Part Hierarchy**: To specify the type-hint for all sub-parts of a part, simply specify the type-hint for the parent part. For example, if all the legs under `legs` are to be fixed, then simply add "legs" : "keep_fixed" to the list. Please only use the hierarchies present in the shape specification. Do not use the symmetry name inside the parentheses (for instance `legs (rotational symmetry)` should be referred to as `legs` only).

6. **Enter Keep Fixed if Specified**: If the user specifies to keep the rest of the parts fixed, add the type hint "keep_fixed" for all the remaining parts. Remember that the "keep_fixed" type hint must be included when the user explicitly specifies it.

7. **Keeping count fixed**: To retain the count of instances of some parts, use the type hint "change_delta" on its parent part. For example, if the user wants to keep the number of drawers fixed, use "drawers" : "change_delta" in the dictionary.

Now, please specify the `type_hints` dictionary by following the instructions above.
"""


shape_specification_no_cot = """Please specify the `type_hints` for the following object according to the user's edit request.

## Shape Specification

$shape_specification
## User's Edit Request

$edit_request

## Steps for Identifying the type hints

1. **Identify Explicitly stated parts**: Determine what parts are *explicitly* referred to in the edit request. Only these parts should be considered for the dictionary.

2. **Identify the requested edit type**: For the *explicitly* referred parts, analyse the edit request to determine the type-hint provided by the user.

3. **Add to the dictionary**: Now add these parts with their type-hint to the dictionary.

## Guidelines

1. **Code Format**: Specify the dictionary in a single Python code snippet, strictly adhering to the instructions above.

2. **Face/Edge Movement**: If the edit request specifies movement of a subpart of a part, the type hint should be 'scale'.

3. **Part Full Names**: A part's full name is its path from the root of the shape. For e.g., if the `back_slat_left` is under `back_slats`, then full name is "back_slats/back_slat_left". 

4. **Using Part Hierarchy**: To specify the type-hint for all sub-parts of a part, simply specify the type-hint for the parent part. For example, if all the legs under `legs` are to be fixed, then simply add "legs" : "keep_fixed" to the list. Please only use the hierarchies present in the shape specification. Do not use the symmetry name inside the parentheses (for instance `legs (rotational symmetry)` should be referred to as `legs` only).

5. **Enter Keep Fixed if Specified**: If the user specifies to keep the rest of the parts fixed, add the type hint "keep_fixed" for all the remaining parts.

6. **Keeping count fixed**: To retain the count of instances of some parts, use the type hint "change_delta" on its parent part. For example, if the user wants to keep the number of drawers fixed, use "drawers" : "change_delta" in the dictionary.

Now, please specify the `type_hints` dictionary by following the instructions above.
"""


variable_list = [
    "shape_specification",
    "edit_request",
    "shape_class"
]

def get_instructions(nocot=NOCOT, with_examples=WITH_EXAMPLES):
    if nocot:
        ss = shape_specification_no_cot
    else:
        ss = shape_specification
    if with_examples:
        instructions = "\n\n".join([start_instructions, examples, ss])
    else:
        instructions = "\n\n".join([start_instructions, ss])
    return instructions