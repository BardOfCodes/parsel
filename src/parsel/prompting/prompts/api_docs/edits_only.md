```python
# Functions to return unit vectors in different directions in the global coordinate system.
def FrontUnitVector() -> Vector:
    ...

def BackUnitVector() -> Vector:
    ...

def LeftUnitVector() -> Vector:
    ...

def RightUnitVector() -> Vector:
    ...

def UpUnitVector() -> Vector:
    ...

def DownUnitVector() -> Vector:
    ...

class Translate:
    """
    Translate a geometric entity (part, face, or edge) in a given direction.

    Parameters:
    operand (Part|Face|Edge): The geometric entity (part, face, or edge) to be translated.
    direction (Vector): The vector specifying the direction and magnitude of the translation.
    """

class Rotate:
    """
    Rotate a geometric entity (part, face, or edge) around a specified axis.

    Parameters:
    operand (Part|Face|Edge): The entity to be rotated.
    rotation_axis_origin (Vector): The origin point of the rotation axis.
    rotation_axis_direction (Vector): The direction vector of the rotation axis.
    """

class Scale:
    """
    Scale a geometric entity in different dimensions, either expanding or contracting.

    This class allows for scaling along one, two, or three dimensions based on the number of axis direction vectors provided:
    - To scale along a single axis, provide axis_direction_1.
    - To scale along two axes, provide both axis_direction_1 and axis_direction_2.
    - To scale uniformly in all directions (i.e. volume scaling), do not provide any axis_direction vectors.

    Parameters:
    operand (Part|Face|Edge): The entity to be scaled.
    scaling_type (str): The type of scaling, either "expand" or "contract".
    scale_origin (Vector): The origin point of the scaling.
    axis_direction_1 (Vector, optional): A direction vector along the first axis of scaling.
    axis_direction_2 (Vector, optional): A direction vector along the second axis of scaling, used along with axis_direction_1 for a two-dimensional scaling. Must be perpendicular to axis_direction_1.

    Note:
    - Provide axis_direction_1 for single-axis scaling. This is preferable over face translations when object needs to be streched or compressed along a single axis (both ways).
    - Provide both axis_direction_1 and axis_direction_2 for two-dimensional scaling.
    - Do not provide any axis_direction vectors for uniform scaling in all directions.
    """

class Shear:
    """
    Shear a geometric entity in a specified direction along a defined plane.
    If the shear plane origin is the center of the entity, the shear will effect the entity in both the directions along the shear direction.

    Parameters:
    operand (Part|Face|Edge): The entity to be sheared.
    shear_plane_origin (Vector): The origin point of the shear plane.
    shear_plane_direction (Vector): The direction vector of the shear plane.
    shear_direction (Vector): The direction vector of the shearing.

    """

class SymGroupEdit:
    """
    Modify the symmetry group of a geometric entity.

    Parameters:
    operand (Part|Face|Edge): The entity with the symmetry group.
    change_type (str): Type of change - 'count' or 'delta' between symmetry group elements.
    extend_from (str): Specifies expansion method for translation symmetry. Options: "start", "end", "center", "keep_fixed". When extend_from is specified as "start", the symmetry group extends towards the direction of the first instance in the symmetry group. When extend_from is specified as "end", the symmetry group extends towards the direction of the last instance in the symmetry group. When extend_from is specified as "center", the symmetry group extends on both sides of the symmetry group. When extend_from is specified as "keep_fixed", the symmetry group remains of fixed size, while the number of instances change.
    """
```