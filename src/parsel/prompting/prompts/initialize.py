from .settings import NOCOT, WITH_EXAMPLES
from .init_wt_examples import overall_instructions
primary_edit_start = """# Overview

You are a 3D shape editing AI assistant. You are given a 3D $shape_class made of labeled cuboidal parts, and a user's edit request in natural language. Your task is to determine the primary (first) part-level edit necessary to fulfill the user's request. Subsequentyly, an edit propagation system will generate the complete edit based on the primary edit.

## Output Specification

Specify the `primary_edit` variable in python using the API specified below. The `primary_edit` should be specified in a code snippet in the following format:

```python
primary_edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API
## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Specifying the Primary Edit

1. Identify the main part of the shape that needs to be edited based on the user's edit request.

2. Determine whether the entire part requires editing or if modifying a specific face or edge is sufficient.

3. For the identified part, identify the edit class(es) from the API which align with the requested modifications for the chosen part.

4. If multiple edit classes fit the request, evaluate the impact of each on the identified part. After this analysis, choose the edit class that best suits the needs.

5. For the chosen edit class, methodically define each attribute, explaining their values and reasons. Utilize functions like `center`, `face_center`, `edge_center`, `direction`, `RightUnitVector`, etc., to specify these parameters."""

primary_edit_end = """Now, please specify a single primary edit by following the instructions above.
"""

primary_edit_guideline_list = [
    """**Analytical Approach**: Follow the above instructions step-by-step and explain your solution along the way.""",
    """**Code Format**: Specify the primary edit in a executable Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.""",
    """**No Imports**: Do not use any import statements in your code. The functions and classes required to create the primary edit are already available.""",
    """**Function Use**: Employ functions like `center` and `face_center` to define the primary edit precisely. Do not use sympy, numpy etc. to create these variables. The amount of the edit will be controlled by the user interface and must not be specified.""",
    """**Single Part Edit**: Edit only a single part/relation. Other parts will be edited with edit propagation.""",
    """**Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class"). Use one of the parts from the shape specification.""",
    """**Cuboidal Parts**: Remember that all the parts are simple cuboids.""",
    """**Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).""",
    """**Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction."""
]
def get_instructions(nocot=NOCOT, with_examples=WITH_EXAMPLES):
    if with_examples:
        instructions = overall_instructions
    else:
        if nocot:
            selected_guidelines = primary_edit_guideline_list[1:]
        else:
            selected_guidelines = primary_edit_guideline_list
        guideline_str = "## Guidelines\n\n"
        for ind, guideline in enumerate(selected_guidelines):
            if ind == len(selected_guidelines) - 1:
                guideline_str += f"{ind + 1}. {guideline}"
            else:
                guideline_str += f"{ind + 1}. {guideline}\n\n"
        instructions = "\n\n".join([primary_edit_start, guideline_str, primary_edit_end])
    return instructions
