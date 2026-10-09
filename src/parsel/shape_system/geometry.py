from collections import defaultdict
from typing import List, Union
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R
from .math import numeric_limited_eval
from .annotations import sympy_vec_to_str, sympy_annotate_via_dp_vec
from .constants import COMPARISON_THRESHOLD, DOT_THRESHOLD

##################### Atoms #####################
UNIT_X = sp.Matrix(1, 3, np.array([1., 0., 0.], dtype=np.float32))
UNIT_Y = sp.Matrix(1, 3, np.array([0., 1., 0.], dtype=np.float32))
UNIT_Z = sp.Matrix(1, 3, np.array([0., 0., 1.], dtype=np.float32))
ORIGIN = sp.Matrix(1, 3, np.array([0., 0., 0.], dtype=np.float32))

DIRECTION_MAP = {
    "back": -UNIT_Z.copy(),
    "front": UNIT_Z.copy(),
    "right": UNIT_X.copy(),
    "left": -UNIT_X.copy(),
    "up": UNIT_Y.copy(),
    "down": -UNIT_Y.copy(),
}

DIRECTIONS = ["front", "back", "right", "left", "up", "down"]
DIR_NP = sp.Matrix.hstack(*[UNIT_Z, -UNIT_Z, UNIT_X, -UNIT_X, UNIT_Y, -UNIT_Y])


PLANE_MAP = {
    "horizontal_facing_up": (ORIGIN.copy(), UNIT_Y.copy()),
    "horizontal_facing_down": (ORIGIN.copy(), UNIT_Y.copy()),
    "vertical_facing_right": (ORIGIN.copy(), UNIT_X.copy()),
    "vertical_facing_left": (ORIGIN.copy(), -UNIT_X.copy()),
    "vertical_facing_back": (ORIGIN.copy(), -UNIT_Z.copy()),
    "vertical_facing_front": (ORIGIN.copy(), UNIT_Z.copy()),
}
NORMAL_DIR_TO_PLANE_MAPPER = {
    "front": "vertical_facing_front",
    "back": "vertical_facing_back",
    "right": "vertical_facing_right",
    "left": "vertical_facing_left",
    "up": "horizontal_facing_up",
    "down": "horizontal_facing_down",
}


LINE_MAP = {
    'left-to-right': (ORIGIN.copy(), UNIT_X.copy()),
    'front-to-back': (ORIGIN.copy(), -UNIT_Z.copy()),
    'down-to-up': (ORIGIN.copy(), UNIT_Y.copy()),
    'right-to-left': (ORIGIN.copy(), -UNIT_X.copy()),
    'back-to-front': (ORIGIN.copy(), UNIT_Z.copy()),
    'up-to-down': (ORIGIN.copy(), -UNIT_Y.copy()),
}

LINE_DIRECTION_TO_LINE_MAPPER = {
    'front': 'back-to-front',
    'back': 'front-to-back',
    'left': 'right-to-left',
    'right': 'left-to-right',
    'up': 'down-to-up',
    'down': 'up-to-down',
}
    
def Vector(value):
    return sp.Matrix(value)

# Ax
class Plane:
    def __init__(self, point=None, normal=None):
        point = sp.Matrix(1, 3, point)
        normal = sp.Matrix(1, 3, normal)
        
        self.normal = normal.normalized()
        # if h is negative, then the plane is facing the opposite direction
        # h = self.normal.dot(point - ORIGIN)
        # self.h = h
        self.point = point
    @property
    def h(self):
        return self.normal.dot(self.point - ORIGIN)
    
    @property
    def name(self):
        name = f"h={self.h}, normal={self.normal[:, 0]}"
        return name
    
    @property
    def origin(self):
        return self.point
    
    def __repr__(self):
        origin_str = sympy_vec_to_str(self.origin)
        normal_str = sympy_annotate_via_dp_vec(self.normal)
        sig = f"Plane(origin={origin_str}, normal={normal_str})"
        return sig
    
    def signature(self):
        return str(self)
    
    def prompt_signature(self):
        normal_str = sympy_annotate_via_dp_vec(self.normal)
        sig = f"Plane facing the {normal_str} direction"
        return sig
    
    def full_signature(self):
        return self.signature()
    
    def copy(self):
        return Plane(origin=self.origin.copy(), normal=self.normal.copy())
    
    def __eq__(self, other):
        if isinstance(other, Plane):
            point_distance = self.point_on_plane(other.origin) and other.point_on_plane(self.origin)
            normal_distance = sp.Abs(self.normal.dot(other.normal)) > DOT_THRESHOLD
            # TODO: What if this is an expression?
            return point_distance and normal_distance
        else:
            return False
    
    def point_on_plane(self, point):
        valid = (point - self.origin).dot(self.normal) < COMPARISON_THRESHOLD
        return valid

class Line:
    def __init__(self, point=None, direction=None):
        point = sp.Matrix(1, 3, point)
        direction = sp.Matrix(1, 3, direction)
        self.direction = direction.normalized()
        self.point = point - (point - ORIGIN).dot(self.direction) * self.direction
    
    def __repr__(self):
        point_string = sympy_vec_to_str(self.point)
        direction_string = sympy_annotate_via_dp_vec(self.direction)
        name = f"Line(point={point_string}, direction={direction_string})"
        return name
    
    def prompt_signature(self):
        raise NotImplementedError
    
    def signature(self):
        sig = str(self)
        return sig
    
    def full_signature(self):
        return self.signature()
    
    def copy(self):
        return Line(point=self.point.copy(), direction=self.direction.copy())
    
    def __eq__(self, other):
        if isinstance(other, Line):
            # directions are aligned
            direction_distance = self.direction.cross(other.direction).norm() < COMPARISON_THRESHOLD
            # points lie in opposite lines
            point_distance = self.point_on_line(other.point) and other.point_on_line(self.point)
            return point_distance and direction_distance
        else:
            return False
    
    def point_on_line(self, point):
        valid = (point - self.point).cross(self.direction).norm() < COMPARISON_THRESHOLD
        return valid

    
def FrontUnitVector():
    return Vector(DIRECTION_MAP["front"])
def BackUnitVector():
    return Vector(DIRECTION_MAP["back"])
def LeftUnitVector():
    return Vector(DIRECTION_MAP["left"])
def RightUnitVector():
    return Vector(DIRECTION_MAP["right"])
def UpUnitVector():
    return Vector(DIRECTION_MAP["up"])
def DownUnitVector():
    return Vector(DIRECTION_MAP["down"])

def HorizontalFacingUpPlane():
    return Plane(*PLANE_MAP["horizontal_facing_up"])
def HorizontalFacingDownPlane():
    return Plane(*PLANE_MAP["horizontal_facing_down"])
def VerticalFacingRightPlane():
    return Plane(*PLANE_MAP["vertical_facing_right"])
def VerticalFacingLeftPlane():
    return Plane(*PLANE_MAP["vertical_facing_left"])
def VerticalFacingBackPlane():
    return Plane(*PLANE_MAP["vertical_facing_back"])
def VerticalFacingFrontPlane():
    return Plane(*PLANE_MAP["vertical_facing_front"])

def LeftToRightLine():
    return Line(*LINE_MAP["left-to-right"])
def FrontToBackLine():
    return Line(*LINE_MAP["front-to-back"])
def DownToUpLine():
    return Line(*LINE_MAP["down-to-up"])
def RightToLeftLine():
    return Line(*LINE_MAP["right-to-left"])
def BackToFrontLine():
    return Line(*LINE_MAP["back-to-front"])
def UpToDownLine():
    return Line(*LINE_MAP["up-to-down"])


    
        
        
        
############ Deprecated ############

class LineSegment:
    def __init__(self, point_start=None, point_end=None):
        self.direction = (point_end - point_start).normalized()
        self.point_start = point_start
        self.point_end = point_end
    
    def __repr__(self):
        point_string_start = sympy_vec_to_str(self.point_start)
        point_string_end = sympy_vec_to_str(self.point_end)
        name = f"LineSegment(point_start={point_string_start}, point_end={point_string_end})"
        return name
    
    def prompt_signature(self):
        raise NotImplementedError
    
    def signature(self):
        sig = str(self)
        return sig
    
    def full_signature(self):
        return self.signature()
    
    def copy(self):
        return LineSegment(point_start=self.point_start.copy(), point_end=self.point_end.copy())
    
    def __eq__(self, other):
        if isinstance(other, Line):
            # directions are aligned
            # points lie in opposite lines
            point_start_distance = (self.point_start - other.point_start).norm() < COMPARISON_THRESHOLD
            point_end_distance = (self.point_end - other.point_end).norm() < COMPARISON_THRESHOLD
            return point_start_distance and point_end_distance
        else:
            return False
