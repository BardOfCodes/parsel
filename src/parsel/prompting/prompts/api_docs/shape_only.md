```python

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
            >>> shape = Part(...)
            >>> leg_part = shape.get("leg_front_right")
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
```