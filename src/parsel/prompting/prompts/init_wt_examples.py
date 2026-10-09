import random
from string import Template
from .settings import NOCOT, WITH_EXAMPLES

instructions = """# Task

Given a 3D shape description, and a user's edit request in natural language, specify the primary edit using the following Editing API.

## Shape Description

The input shape will be composed of 3D Cuboids with semantic annotation. You will be given a shape-hierarchy tree, which indicates the relationship between the cuboids. To access a particular part or its features (such as a face or an edge) use to following API:

$SHAPE_API

## Edit API

The Edit API helps specify part-level edits. You must specify the edit command in a python snippet. Since the current task is to only specify the primary edit, you will only need to specify one edit command. Refer to the following API to specify the edit command:

$EDIT_API

## Output Specification

Specify the `primary_edit` variable in python using the API specified above. The `primary_edit` should be specified in a code snippet in the following format:

```python
primary_edit = ... # must be created via the API
```
"""

examples_template = """## Examples 

To help understand the task better, here are a few examples:
Note that part names must be dependant on the shape specification provided in the prompt. In the following examples we only provide the edit request and the expected primary edit.

$examples
"""

post_examples = """## Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explaining your solution along the way.

2. **Code Format**: Specify the primary edit in a executable Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.

3. **No Imports**: Do not use any import statements in your code. The functions and classes required to create the primary edit are already available.

4. **Function Use**: Employ functions like `center` and `face_center` to define the primary edit precisely. Do not use sympy, numpy etc. to create these variables. The amount of the edit will be controlled by the user interface and must not be specified

5. **Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class"). Use one of the parts from the shape specification.

6. **Cuboidal Parts**: Remember that all the parts are simple cuboids.

7. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).

8. **Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction.

## Current Task

The shape is described as follows:

$shape_specification
The user has the following natural language request: $edit_request

Please specify the primary edit.
"""


post_examples_no_cot = """## Guidelines

1. **Code Format**: Specify the primary edit in a executable Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.

2. **No Imports**: Do not use any import statements in your code. The functions and classes required to create the primary edit are already available.

3. **Function Use**: Employ functions like `center` and `face_center` to define the primary edit precisely. Do not use sympy, numpy etc. to create these variables. The amount of the edit will be controlled by the user interface and must not be specified

4. **Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class"). Use one of the parts from the shape specification.

5. **Cuboidal Parts**: Remember that all the parts are simple cuboids.

6. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).

7. **Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction.

## Current Task

The shape is described as follows:

$shape_specification
The user has the following natural language request: $edit_request

Please specify the primary edit.
"""

example_1 = """Edit Request: "I want to raise the seat of the chair upwards."

Primary Edit

```python
part =  shape.get("seat_surface")
primary_edit = Translate(seat, direction=UpUnitVector())
```"""

example_2 = """Edit Request: "I want move the front legs of the chair backwards."

Primary Edit

```python
leg_front_left = shape.get("legs/leg_front_left")
edit_1 = Translate(leg_front_left, direction=BackUnitVector())
leg_front_right = shape.get("legs/leg_front_right")
edit_2 = Translate(leg_front_right, direction=BackUnitVector())
primary_edit = edit_1 # or edit_2; only one of them should be declared as the primary edit.
```"""

example_3 = """Edit Request: "I want to make the seat thicker from below."

Primary Edit

```python
seat = shape.get("seat")
primary_edit = Translate(seat.face("down"), direction=DownUnitVector())
```"""

example_4 = """Edit Request: "I want to make the bed wider on the right side."

Primary Edit

```python
# pick a central part of the bed to make it wider
bed_frame = shape.get("mattress")
primary_edit = Translate(bed_frame.face("right"), direction=RightUnitVector())
```"""

example_5 = """Edit Request: "I want to shorten the legs from the top."

Primary Edit

```python
# Shorten any of the legs
leg_front_left = shape.get("legs/leg_front_left")
primary_edit = Translate(leg_front_left.face("up"), direction=DownUnitVector())
```"""

example_6 = """Edit Request: "I want to tilt the armrests downwards."

Primary Edit

```python
part = shape.get("arm_rest_horizontal_bar_left") # tilt the horizontal component of the armrest.
# If the edit request does not explicitly request rotation, prefer translating a face.
primary_edit = Translate(part.face("front"), direction=DownUnitVector())
```"""

example_7 = """Edit Request: "I want to move the base of the armrests inwards."

Primary Edit

```python
part = shape.get("arm_rest_vertical_bar_right") 
primary_edit = Translate(part.face("down"), direction=LeftUnitVector())
```"""

example_8 = """Edit Request: "I want to extend the top right side of the benches backrest towards the sky."

Primary Edit

```python
part = shape.get("back")
primary_edit = Translate(part.edge("top", "right"), direction=UpUnitVector())
```"""

example_9 = """Edit Request: "I want to shift the foot of front legs forward without moving its top."

Primary Edit

```python
part = shape.get("legs/leg_front_right")
primary_edit = Translate(part.face("down"), direction=FrontUnitVector())
```"""

example_10 = """Edit Request : "I want to tilt the backrest of the bed further to the back."

Primary Edit

```python
part = shape.get("headboard")
# The origin should the the center of the bottom face of the backrest.
rotation_origin = part.face_center("down")
# Using the right-hand rule, to rotate the backrest further to the back, the axis of rotation should be the towards the left.
rotation_axis = LeftUnitVector()
primary_edit = Rotate(bed_back_frame, rotation_origin, rotation_axis)
```"""

example_11 = """Edit Request: "I want to rotate the armrests downwards."

Primary Edit

```python
# rotate the horizontal component of the armrest.
part = shape.get("arm_rest_horizontal_bar_left") 
# The origin will be the face center of the back face.
rotation_origin = part.face_center("back")
# The direction using right hand rule should be towards the right.
rotation_direction = RightUnitVector()
primary_edit = Rotate(part, rotation_origin, rotation_direction)
```"""

example_12 = """Edit Request: "I want to lengthen the table along the left-right direction."

Primary Edit

```python
part = shape.get("table_top")
scaling_type = "expand"
scaling_origin = table_top.center()
scaling_direction = LeftUnitVector()
primary_edit = Scale(part, scaling_type, scaling_origin, scaling_direction)
```"""

example_13 = """Edit Request: "I want to make the legs radially thicker."

Primary Edit

```python
part = shape.get("leg_front_right")
scaling_type = "expand"
scale_origin = part.center()
scale_direction_1 = FrontUnitVector()
scale_direction_2 = RightUnitVector()
primary_edit = Scale(part, scaling_type, scale_origin, scale_direction_1, scale_direction_2)
```"""

old_example_13 = """Edit Request: "I want to reduce the width of the top of the lampshade along the front-back direction."

Primary Edit

```python
part = shape.get("lampshade")
scaling_type = "contract"
scale_origin = part.center()
scale_direction = FrontUnitVector()
primary_edit = Scale(part, scaling_type, scale_origin, scale_direction)
```"""

example_14 = """Edit Request: "I want to make the table wider radially."

Primary Edit

```python
part = shape.get("table_top")
table_top_center = part.center()
# Expand along 2 axes to make the table wider radially.
scaling_type = "expand"
scaling_origin = table_top_center
axis_direction_1 = LeftUnitVector()
axis_direction_2 = FrontUnitVector()
primary_edit = Scale(part, scaling_type, scaling_origin, axis_direction_1, axis_direction_2)
```"""

example_15 = """Edit Request: "I want to make the armrests smaller in front-back and up_down directions while remaining attached to the back as it is."

Primary Edit

```python
part = shape.get("arm_rest_horizontal_bar_left")
# As the armrests are attached to the back, the scaling origin should be the back top edge center.
scaling_type = "contract"
scaling_origin = part.edge_center("top", "back")
axis_direction_1 = FrontUnitVector()
axis_direction_2 = UpUnitVector()
primary_edit = Scale(part, scaling_type, scaling_origin, axis_direction_1, axis_direction_2)
```"""

example_16 = """Edit Request: I want to make the lampshade bigger in all directions.

Primary Edit

```python
part = shape.get("lampshade")
# Lets scale about the bottom face center of the lampshade.
scaling_type = "expand"
scaling_origin = part.face_center("down")
primary_edit = Scale(part, scaling_type, scaling_origin)
```"""

example_17 = """Edit Request: "I want to make the front of the table wider."

Primary Edit

```python
part = shape.get("table_top")
scaling_type = "expand"
scaling_origin = part.face_center("front")
scaling_direction = RightUnitVector()
primary_edit = Scale(part.face("front"), scaling_type, scaling_origin, scaling_direction)
```"""

example_18 = """Edit Request: "I want to move the front of the seat upwards, and the back of the seat downwards."

Primary Edit

```python
seat = shape.get("seat")
shear_center = seat.center()
shear_direction = UpUnitVector()
shear_plane_normaal = FrontUnitVector()
primary_edit = Shear(seat, shear_direction, shear_center, shear_plane_normal)
```"""

example_19 = """Edit Request: "I want to the make the bed slanted vertically towards the right."

Primary Edit

```python
sideboards = shape.get("sideboards")
shear_center = sideboards.edge_center("top", "right") 
shear_direction = RightUnitVector()
shear_plane_normal = UpUnitVector()
primary_edit = Shear(sideboards, shear_direction, shear_center, shear_plane_normal)
```"""

example_20 = """Edit Request: "I want to increase the number of legs of the chair."

Primary Edit

```python
legs = shape.get("legs")
primary_edit = SymGroupEdit(legs, change_type="count", extend_from="keep_fixed")
```"""

example_21 = """Edit Request: "I want to adjust the spacing between the back slats."

Primary Edit

```python
back_vertical_bars = shape.get("back_vertical_bars")
primary_edit = SymGroupEdit(back_vertical_bars, change_type="delta", extend_from="center")
```"""

examples = [example_1, example_2, example_3, example_4, example_5, example_6, example_7, example_8, example_9, example_10, example_11, example_12, example_13, example_14, example_15, example_16, example_17, example_18, example_19, example_20, example_21]

random.shuffle(examples)
examples = [f"### Example {ind + 1}\n\n{example}" for ind, example in enumerate(examples)]
examples_str = "\n\n".join(examples)
examples = Template(examples_template).substitute(examples=examples_str)
if NOCOT:
    overall_instructions = "\n\n".join([instructions, examples, post_examples_no_cot])
else:
    overall_instructions = "\n\n".join([instructions, examples, post_examples])