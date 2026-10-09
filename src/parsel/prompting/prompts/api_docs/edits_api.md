```python

from typing import Union

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

DIRECTION_STRINGS = ["front", "back", "left", "right", "up", "down"]

class Part:
    """A class representing a part of a shape, capable of containing nested sub-parts."""

    def get(self, label: str) -> Part:
        """
        Retrieve a sub-part using its label.

        Parameters:
        label (str): The label of the sub-part.

        Returns:
        Part: The sub-part with the specified label.

        Example:
            >>> leg_part = shape.get("leg_front_right")
            >>> # Access nested parts as follows:
            >>> central_back_slat = shape.get("back_slats/back_slat_center")
        """

    def center(self) -> Vector:
        """
        Returns the center point of this part's geometry in global coordinates.

        Returns:
        Vector: The center point of the part.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_center = seat.center()
        """

    def face_center(self, direction: str) -> Vector:
        """
        Returns the center of a face of this part in a given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction (str): The direction of the face.

        Returns:
        Vector: The center of the specified face.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_front_center = seat.face_center("front")
        """

    def edge_center(self, direction_1: str, direction_2: str) -> Vector:
        """
        Returns the center of an edge formed by two intersecting faces of this part.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1 (str): The direction of the first intersecting face.
        direction_2 (str): The direction of the second intersecting face.

        Returns:
        Vector: The center of the edge at the intersection.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_back_right_edge_center = seat.edge_center("back", "right")
        """

    def corner(self, direction_1: str, direction_2: str, direction_3: str) -> Vector:
        """
        Returns the corner where three faces of this part intersect.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1, direction_2, direction_3 (str): Directions of the intersecting faces.

        Returns:
        Vector: The corner at the intersection of the specified faces.

        Example:
            >>> back_bars = shape.get("back_surface_vertical_bars")
            >>> back_bars_corner = back_bars.corner("front", "right", "up")
        """

    def direction(self, direction: str) -> Vector:
        """
        Returns the vector connecting the center of this part to the center of a face in a given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction (str): The direction of the face.

        Returns:
        Vector: The vector pointing from the center to the center of the specified face.
        """

    def face(self, name: str) -> Face:
        """
        Retrieve the face of this part in the given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        name (str): The name or direction of the face.

        Returns:
        Face: The face in the specified direction.

        Example:
            >>> seat = shape.get("seat")
            >>> front_seat_face = seat.face("front")
            >>> bed_frame = shape.get("bed_frame")
            >>> bottom_bed_face = bed_frame.face("down")
        """

    def edge(self, direction_1: str, direction_2: str) -> Edge:
        """
        Obtain the edge of this part along two specified directions.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1, direction_2 (str): The directions defining the edge.

        Returns:
        Edge: The edge along the specified directions.

        Example:
            >>> seat = shape.get("legs/leg_front_right")
            >>> back_right_seat_edge = seat.edge("back", "right")
        """
        
## Edits
type_operand = Union[Part, Face, Edge]

class Translate:
    """
    Translate a geometric entity (part, face, or edge) in a given direction.

    Parameters:
    operand (Part|Face|Edge): The geometric entity (part, face, or edge) to be translated.
    direction (Vector): The vector specifying the direction and magnitude of the translation.

    Examples:
        # Raise the seat of a chair.
        seat = shape.get("seat_surface")
        edit = Translate(seat, direction=UpUnitVector())

        # Move the front legs of the chair backwards.
        leg_front_left = shape.get("legs/leg_front_left")
        edit_1 = Translate(leg_front_left, direction=BackUnitVector())
        leg_front_right = shape.get("legs/leg_front_right")
        edit_2 = Translate(leg_front_right, direction=BackUnitVector())

        # Lower the backrest of the chair along the vertical bars in contact with it.
        backrest = shape.get("backrest")
        back_vertical_bar = shape.get("back_surface_vertical_bar_left")
        edit = Translate(backrest, direction=back_vertical_bar.direction("down"))

        # Translate a face to scale a part from one side.
        # Make the seat thicker by translating its bottom face downwards.
        seat = shape.get("seat_surface")
        edit = Translate(seat.face("down"), direction=DownUnitVector())

        # Stretch a part in one direction by translating a face.
        bed_frame = shape.get("bed_frame")
        edit = Translate(bed_frame.face("right"), direction=RightUnitVector())

        # Shorten the front legs of the chair from the top.
        leg_front_left = shape.get("legs/leg_front_left")
        edit = Translate(leg_front_left.face("up"), direction=DownUnitVector())

        # Move the base of a part: Move the base of the armrests inwards.
        armrest_left = shape.get("armrest_left")
        edit = Translate(armrest_left.face("down"), direction=RightUnitVector())

        # Tilt the front of a chair's armrest downwards.
        armrest_left = shape.get("armrest_left")
        edit = Translate(armrest_left.face("front"), direction=DownUnitVector())
        
        # Taper a part in one direction by translating an edge.
        # Extend only the top right side of the chair's back.
        back = shape.get("back_surface")
        edit = Translate(back.edge("top", "right"), direction=RightUnitVector())

        # Shear a part by translating a face while keeping another part stationary.
        # Move the base of the front left leg forward while keeping its attachment to the seat unchanged.
        leg_front_left = shape.get("legs/leg_front_left")
        edit = Translate(leg_front_left.face("down"), direction=FrontUnitVector())
    """

class Rotate:
    """
    Rotate a geometric entity (part, face, or edge) around a specified axis.

    Parameters:
    operand (Part|Face|Edge): The entity to be rotated.
    rotation_axis_origin (Vector): The origin point of the rotation axis.
    rotation_axis_direction (Vector): The direction vector of the rotation axis.

    Examples:
        # Rotate the back frame of the bed backwards
        bed_back_frame = shape.get("bed_frame_back")
        rotation_origin = bed_back_frame.face_center("bottom")
        # Using the right-hand-rule, to rotate backwards, the axis must point towards the left.
        rotation_direction = LeftUnitVector()
        # To rotate towards the back, the axis of rotation is the left unit vector.
        edit = Rotate(bed_back_frame, rotation_origin, rotation_direction)

        # Tilt the armrests of the chair downwards
        armrest_left = chair.get("armrest_left")
        rotation_origin = armrest_left.face_center("back")
        # using the right-hand-rule to tile the armrest downwards, the axis must point towards the right.
        rotation_direction = RightUnitVector()
        edit = Rotate(backrest, rotation_origin, rotation_direction)
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

    Examples:
        # Scaling along a single axis

        # Lengthen the table top along the left-right direction
        table_top = shape.get("table_top")
        scaling_type = "expand"
        scale_origin = table_top.center()
        scale_direction = RightUnitVector()
        edit = Scale(table_top, scaling_type, scale_origin, scale_direction)

        # Reduce the width of the front of a lamp head.
        lamp_head = shape.get("lamp_head")
        scaling_type = "contract"
        scale_origin = lamp_head.center()
        scale_direction = RightUnitVector()
        edit = Scale(lamp_head.face("front"), scale_type, scale_origin, scale_direction)

        # Scaling along two axes

        # Bring the table legs closer together, given the table is in upright position.
        table_legs = shape.get("legs")
        table_legs_center = table_legs.center()
        scaling_type = "contract"
        scale_origin = table_legs_center
        scale_direction_1 = FrontUnitVector()
        scale_direction_2 = RightUnitVector()
        edit = Scale(table_legs, scaling_type, scale_origin, scale_direction_1, scale_direction_2)

        # Scale down the armrests along both the back-front and up-down directions
        armrest_left = shape.get("armrest_left")
        scaling_type = "contract"
        scale_origin = armrest_left.edge_center("down", "back")
        scale_direction_1 = DownUnitVector()
        scale_direction_2 = BackUnitVector()
        edit = Scale(armrest_left, scaling_type, scale_origin, scale_direction_1, scale_direction_2)

        # Uniform scaling in all directions

        # Shrink a lamp cover uniformly from its top
        lamp_cover = shape.get("lamp_cover")
        scaling_type = "contract"
        scale_center = lamp_cover.face_center("up")
        edit = Scale(lamp_cover, scaling_type, scale_origin)
        
        # Make a side of a part wider. 

        # make the front of the chair seat wider.
        seat = shape.get("seat_surface")
        scaling_type = "expand"
        scale_origin = seat.face_center("back")
        scale_direction = RightUnitVector()
        edit = Scale(seat.face("front"), scaling_type, scale_origin, scale_direction)

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
    shear_direction (Vector): The direction vector of the shearing.
    shear_plane_origin (Vector): The origin point of the shear plane.
    shear_plane_direction (Vector): The direction vector of the shear plane.

    Examples:
        # Shear the backrest of a chair, bringing the top forward and the bottom backward.
        backrest = shape.get("backrest")
        shear_diretion = RightUnitVector()
        shear_center = backrest.center()
        shear_plane_normal = UpUnitVector()
        edit = Shear(backrest, shear_direction, shear_center, shear_plane_normal)

        # Shear the bed to make it slanted vertically
        bed_frame = shape.get("bed_frame")
        shear_direction = RightUnitVector()
        shear_plane_origin = bed_frame.get_edge_center("left", "down")
        shear_plane_direction = FrontUnitVector()
        edit = Shear(bed_frame, shear_direction, shear_plane_origin, shear_plane_direction)
    """

class SymGroupEdit:
    """
    Modify the symmetry group of a geometric entity.

    Parameters:
    operand (Part|Face|Edge): The entity with the symmetry group.
    change_type (str): Type of change - 'count' or 'delta' between symmetry group elements.
    extend_from (str): Specifies expansion method for translation symmetry. Options: "start", "end", "center", "keep_fixed". When extend_from is specified as "start", the symmetry group extends towards the direction of the first instance in the symmetry group. When extend_from is specified as "end", the symmetry group extends towards the direction of the last instance in the symmetry group. When extend_from is specified as "center", the symmetry group extends on both sides of the symmetry group. When extend_from is specified as "keep_fixed", the symmetry group remains of fixed size, while the number of instances change.

    Examples:
        # Change the number of legs in a swivel chair - given they are in a rotational symmetry group.
        legs = shape.get("legs")
        edit = SymGroupEdit(legs, change_type="count", extend_from="keep_fixed")

        # Adjust the spacing between the vertical bars in the back of a chair - given they are in a translational symmetry group.
        back_vertical_slats = shape.get("back_surface_vertical_bars")
        edit = SymGroupEdit(back_vertical_slats, change_type="delta", extend_from="center")
    """

```
