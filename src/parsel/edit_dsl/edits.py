import numpy as np
import sympy as sp
import os
import re
from typing import List, Tuple, Union as type_union
from ..shape_system.geometry import Line, Plane
from ..shape_system.shape import PrimitiveFeature, Primitive, Part
from ..shape_system.math import evaluate_equals_zero, rotation_matrix_sympy
from ..shape_system.annotations import round_expr
from ..shape_system.geometry import COMPARISON_THRESHOLD, DIRECTIONS as directions
from ..shape_system.constants import PART_ACTIVE, PART_INACTIVE
MAIN_VAR = sp.Symbol("X")
ZERO_VECTOR = sp.Matrix(1, 3, [0, 0, 0])
OP_TYPE = type_union[Part, Primitive, PrimitiveFeature]
DIR_NAME = ["right", "up", "front", "left", "down", "back"]


def get_direction_string(direction, primitive):
    names = []
    points = []
    for key, value in primitive.name_to_indices.items():
        if isinstance(key, str):
            face_center = primitive.face_center(key)
            face_dir = face_center - primitive.center()
            face_dir = face_dir.normalized()
            points.append(np.array(face_dir).astype(np.float32))
            names.append(key)
        elif len(key) == 2:
            edge_center = primitive.edge_center(*key)
            edge_dir = edge_center - primitive.center()
            edge_dir = edge_dir.normalized()
            points.append(np.array(edge_dir).astype(np.float32))
            name = f"{key[0]}-{key[1]}"
            names.append(name)
        elif len(key) == 3:
            corner = primitive.corner(*key)
            corner_dir = corner - primitive.center()
            corner_dir = corner_dir.normalized()
            points.append(np.array(corner_dir).astype(np.float32))

            name = f"{key[0]}-{key[1]}-{key[2]}"
            names.append(name)
    points = np.concatenate(points, axis=0)
    dir_np = np.array(direction).astype(np.float32)
    dot_prods = (points * dir_np).sum(axis=1)

    max_dot_prod = max(dot_prods)
    max_index = np.argmax(dot_prods)
    return names[max_index]


def get_amount_str(expression):
    rounded_expr = round_expr(expression)
    amount_str = f"an amount of {rounded_expr}"
    return amount_str


def get_bidirection_string(direction, primitive):
    names = []
    points = []
    for key, value in primitive.name_to_indices.items():
        if isinstance(key, str):
            face_center = primitive.face_center(key)
            face_dir = face_center - primitive.center()
            face_dir = face_dir.normalized()
            points.append(np.array(face_dir).astype(np.float32))
            names.append(key)
        elif len(key) == 2:
            edge_center = primitive.edge_center(*key)
            edge_dir = edge_center - primitive.center()
            edge_dir = edge_dir.normalized()
            points.append(np.array(edge_dir).astype(np.float32))
            name = f"{key[0]}-{key[1]}"
            names.append(name)
        elif len(key) == 3:
            corner = primitive.corner(*key)
            corner_dir = corner - primitive.center()
            corner_dir = corner_dir.normalized()
            points.append(np.array(corner_dir).astype(np.float32))

            name = f"{key[0]}-{key[1]}-{key[2]}"
            names.append(name)
    points = np.concatenate(points, axis=0)
    dir_np = np.array(direction).astype(np.float32)
    dot_prods = (points * dir_np).sum(axis=1)

    max_index = np.argmax(dot_prods)
    to_name = names[max_index]

    min_index = np.argmin(dot_prods)
    from_name = names[min_index]
    bidir_string = f"{from_name} to {to_name}"

    return bidir_string


def get_point_string(point, operand, shape):
    all_interesting_points, all_point_names = shape.get_all_interesting_points(
        with_names=True)
    primitive_points, primitive_point_names = operand.primitive.get_all_interesting_points(
        with_names=True)
    primitive_points = np.concatenate(primitive_points, axis=0)
    all_points = np.concatenate(
        [all_interesting_points, primitive_points], axis=0)
    all_point_names = all_point_names + primitive_point_names
    point = np.array(point).astype(np.float32)
    distances = np.linalg.norm(all_points - point, axis=1)
    min_index = np.argmin(distances)
    return all_point_names[min_index]


def get_reflected_point(point, plane):
    point_vec = point - plane.origin
    ref_point_vec = point_vec - 2 * point_vec.dot(plane.normal) * plane.normal
    # ref_point_vec = ref_point_vec.normalized()
    ref_point_vec = ref_point_vec + plane.origin
    return ref_point_vec


def get_reflected_vector(dir_vec, plane):
    reflected_vec = dir_vec - 2 * dir_vec.dot(plane.normal) * plane.normal
    ref_vec = reflected_vec.normalized()
    return ref_vec


def get_rotated_point(point, origin, rot_vec, angle):
    point_vec = point - origin
    rotated_pt_vec = point_vec * sp.cos(angle) + rot_vec.cross(point_vec) * sp.sin(
        angle) + rot_vec * rot_vec.dot(point_vec) * (1 - sp.cos(angle))
    rotated_pt_vec = rotated_pt_vec + origin
    return rotated_pt_vec


def get_rotated_vector(dir_vec, rot_vec, angle):
    rotated_vec = dir_vec * sp.cos(angle) + rot_vec.cross(dir_vec) * sp.sin(
        angle) + rot_vec * rot_vec.dot(dir_vec) * (1 - sp.cos(angle))
    rotated_vec = rotated_vec.normalized()
    return rotated_vec


def get_translated_point(point, delta):
    point_vec = point + delta
    return point_vec


def get_translated_vector(vector, delta):
    return vector

# Need function to annotate point, and to annotate direction.


class Edit:
    def __init__(self, operand, amount, identifier):
        self.operand = operand
        if amount is None:
            # set it based on deformation.
            self.amount = MAIN_VAR
        else:
            self.amount = amount
        self.param_names = []
        if identifier is None:
            identifier = os.urandom(16).hex()
        self.identifier = identifier
        self.edit_type = self.__class__.__name__

    @property
    def params(self):
        return {x: getattr(self, x) for x in self.param_names}

    @property
    def name(self):
        name_str = self.__class__.__name__
        name = re.sub('([A-Z]+)', r'_\1', name_str).lower()
        return name

    def propagate(self):
        # Just attach to the primitive
        # TODO: How should this be changed?
        if isinstance(self.operand, Part):
            prim_set = [self.operand.primitive]
        elif isinstance(self.operand, Primitive):
            prim_set = [self.operand]
        for primitive in prim_set:
            primitive.edit_sequence.append(self)

    def __repr__(self):
        param_str_list = [f"{x}={y}" for x, y in self.params.items()]
        all_params = [f"operand={self.operand}"] + param_str_list
        param_str = ', '.join([x for x in all_params])
        string = f"{self.__class__.__name__}({param_str})"
        return string

    # Secondary
    def signature(self , something):
        ...

    def full_signature(self):
        ...

    def code_signature(self):
        ...

    def prompt_signature(self):
        ...

    @staticmethod
    def _prompt_signature(*args, **kwargs):
        ...

    def save_format(self):
        if isinstance(self.operand, Part):
            operand_type = "part"
            index = self.operand.part_index
        else:
            operand_type = "relation"
            index = self.operand.relation_index
        edit_gen = (EditGen(self.__class__, param_dict=self.params,
                    amount=self.amount), operand_type, index)

        return edit_gen

    def copy(self, operand):
        return self.__class__(operand=operand, amount=self.amount, **self.params)


class PartEdit(Edit):
    ...


class RestrictedEdit(PartEdit):

    def get_restrictor_name(self, name):
        if not isinstance(name, str):
            if len(name) == 1:
                name = name[0]
            else:
                name = tuple(sorted(list(name)))
        return name


class PartTranslate(PartEdit):
    def __init__(self, operand: OP_TYPE, direction: sp.Matrix, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.direction = sp.Matrix(1, 3, direction).normalized()
        self.param_names = ["direction"]

    def apply(self, geometric_expr):
        dir_stacked = sp.Matrix.vstack(
            *[self.direction,]*geometric_expr.shape[0])
        geometric_expr = geometric_expr + dir_stacked * self.amount
        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_vec = get_reflected_vector(self.direction, plane)
        new_edit = PartTranslate(
            operand=operand, direction=ref_vec, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        rot_vec = get_rotated_vector(self.direction, rot_vec, angle)
        new_edit = PartTranslate(
            operand=operand, direction=rot_vec, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        new_edit = PartTranslate(
            operand=operand, direction=self.direction, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartTranslate):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            direction_equal = self.direction.cross(
                edit.direction).norm() < COMPARISON_THRESHOLD
            valid = direction_equal and amount_equal
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartTranslate):
            delta = (self.direction * self.amount) - \
                (edit.direction * edit.amount)
            valid = True
            for i in range(3):
                valid = evaluate_equals_zero(
                    delta[i], mode=mode, value=multiplier)
                if not valid:
                    break
        return valid


class RestrictedTranslate(RestrictedEdit):
    def __init__(self, operand: OP_TYPE, direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.direction = sp.Matrix(1, 3, direction).normalized()
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ["direction", "restrictor_name"]
        self.unrestricted_class = PartTranslate

    def apply(self, geometric_expr):
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        dir_list = []
        for i in range(n_points):
            if i in valid_indices:
                dir_list.append(self.direction)
            else:
                dir_list.append(ZERO_VECTOR)
        dir_stacked = sp.Matrix.vstack(*dir_list)
        geometric_expr = geometric_expr + (dir_stacked * self.amount)
        return geometric_expr

    def get_reflected(self, operand, plane, ref_name):
        ref_vec = get_reflected_vector(self.direction, plane)
        new_edit = self.__class__(
            operand=operand, direction=ref_vec, restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        rot_vec = get_rotated_vector(self.direction, rot_vec, angle)
        new_edit = self.__class__(
            operand=operand, direction=rot_vec, restrictor_name=rot_name, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        new_edit = self.__class__(
            operand=operand, direction=self.direction, restrictor_name=trans_name, amount=self.amount)
        return new_edit

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                delta = (self.direction * self.amount) - \
                    (edit.direction * edit.amount)
                valid = True
                for i in range(3):
                    valid = evaluate_equals_zero(
                        delta[i], mode=mode, value=multiplier)
                    if not valid:
                        break
        return valid

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                direction_equal = self.direction.cross(
                    edit.direction).norm() < COMPARISON_THRESHOLD
                valid = direction_equal and amount_equal
        return valid

    def signature(self, shape):
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        center = self.operand.center()
        if self.restrictor_type == "face":
            feature_point = self.operand.face_center(self.restrictor_name)
        else:
            feature_point = self.operand.edge_center(*self.restrictor_name)
        pointing = feature_point - center
        dot = pointing.dot(self.direction)
        if amount_sign == 1:
            dir_str = get_direction_string(self.direction, self.operand)
            amount_str = get_amount_str(self.amount)
            # its expanding if these two are aligned
            scale_str = "Expand"
            if dot < 0:
                scale_str = "Contract"
        else:
            dir_str = get_direction_string(-self.direction, self.operand)
            amount_str = get_amount_str(-self.amount)
            scale_str = "Contract"
            if dot < 0:
                scale_str = "Expand"
        sig = f"{scale_str} {self.operand.full_label} by translating its {self.restrictor_name} {self.restrictor_type} by {amount_str} towards its {dir_str} direction."
        return sig


class PointTranslate(RestrictedTranslate):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, direction: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(PointTranslate, self).__init__(operand=operand, direction=direction,
                                             restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeTranslate(RestrictedTranslate):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, direction: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(EdgeTranslate, self).__init__(operand=operand, direction=direction,
                                            restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceTranslate(RestrictedTranslate):
    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, direction: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(FaceTranslate, self).__init__(operand=operand, direction=direction,
                                            restrictor_name=restrictor_name, amount=amount, identifier=identifier)
        # based on the specs, change edit type
        dir_to_face = self.operand.face_center(
            self.restrictor_name) - self.operand.center()
        trans_dir = self.direction
        dir_to_face = dir_to_face.normalized()
        trans_dir = trans_dir.normalized()
        dot = dir_to_face.dot(trans_dir)
        if np.abs(dot) > np.cos(np.pi/8):
            # kinda like stretching
            self.edit_type = "StretchByFaceTranslate"
        else:
            self.edit_type = "ShearByFaceTranslate"


class PartRotate(PartEdit):

    def __init__(self, operand, rotation_origin, rotation_axis, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.rotation_origin = sp.Matrix(1, 3, rotation_origin)
        self.rotation_axis = sp.Matrix(1, 3, rotation_axis).normalized()
        self.param_names = ['rotation_origin', 'rotation_axis']
        self.rot_matrix = rotation_matrix_sympy(
            self.rotation_axis, self.amount)

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        if len(self.rotation_axis.free_symbols) > 0:
            rot_matrix = rotation_matrix_sympy(self.rotation_axis, self.amount)
        else:
            rot_matrix = self.rot_matrix
        origin_stack = sp.Matrix.vstack(
            *[self.rotation_origin,]*geometric_expr.shape[0])
        geometric_expr = geometric_expr - origin_stack
        geometric_expr = geometric_expr * rot_matrix.T
        geometric_expr = geometric_expr + origin_stack
        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_point_vec = get_reflected_point(self.rotation_origin, plane)
        ref_dir_vec = get_reflected_vector(self.rotation_axis, plane)
        ref_dir_vec = -ref_dir_vec
        new_edit = PartRotate(operand=operand, rotation_axis=ref_dir_vec,
                              rotation_origin=ref_point_vec, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        # Rotation of things in a rotation symmetry is tricky.
        rotated_pt_vec = get_rotated_point(
            self.rotation_origin, origin, rot_vec, angle)
        rotated_dir_vec = get_rotated_vector(
            self.rotation_axis, rot_vec, angle)
        new_edit = PartRotate(operand=operand, rotation_axis=rotated_dir_vec,
                              rotation_origin=rotated_pt_vec, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        shifted_origin = get_translated_point(self.rotation_origin, delta)
        new_edit = PartRotate(operand=operand, rotation_axis=self.rotation_axis,
                              rotation_origin=shifted_origin, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartRotate):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            if amount_equal:
                line_a = Line(point=self.rotation_origin,
                              direction=self.rotation_axis)
                line_b = Line(point=edit.rotation_origin,
                              direction=edit.rotation_axis)
                valid = line_a == line_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartRotate):
            amount_equal = evaluate_equals_zero(
                self.amount - edit.amount, mode=mode, value=multiplier)
            if amount_equal:
                line_a = Line(point=self.rotation_origin,
                              direction=self.rotation_axis)
                line_b = Line(point=edit.rotation_origin,
                              direction=edit.rotation_axis)
                valid = line_a == line_b
        return valid

    def signature(self, shape):
        sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if sign == 1:
            amount_str = get_amount_str(self.amount)
            direction_str = get_direction_string(
                self.rotation_axis, self.operand)
        else:
            amount_str = get_amount_str(-self.amount)
            direction_str = get_direction_string(
                -self.rotation_axis, self.operand)
        origin_string = get_point_string(
            self.rotation_origin, self.operand, shape)
        sig = f"Rotate the {self.operand.full_label} by {amount_str} with rotation axis pointing towards its {direction_str} direction and rotation origin at {origin_string}."
        return sig


class RestrictedRotate(RestrictedEdit):

    def __init__(self, operand: OP_TYPE, rotation_origin: sp.Matrix, rotation_axis: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.rotation_origin = sp.Matrix(1, 3, rotation_origin)
        self.rotation_axis = sp.Matrix(1, 3, rotation_axis).normalized()
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ['rotation_origin',
                            'rotation_axis', 'restrictor_name']

        self.unrestricted_class = PartRotate
        self.rot_matrix = rotation_matrix_sympy(
            self.rotation_axis, self.amount)

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        if len(self.rotation_axis.free_symbols) > 0:
            rot_matrix = rotation_matrix_sympy(self.rotation_axis, self.amount)
        else:
            rot_matrix = self.rot_matrix

        origin_stack = sp.Matrix.vstack(*[self.rotation_origin,]*n_points)
        rot_geometric_expr = geometric_expr - origin_stack
        rot_geometric_expr = rot_geometric_expr * rot_matrix.T
        rot_geometric_expr = rot_geometric_expr + origin_stack

        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        rot_list = []
        for i in range(n_points):
            if i in valid_indices:
                rot_list.append(rot_geometric_expr[i, :])
            else:
                rot_list.append(geometric_expr[i, :])
        rot_stacked = sp.Matrix.vstack(*rot_list)
        return rot_stacked

    def get_reflected(self, operand, plane, ref_name):
        ref_point_vec = get_reflected_point(self.rotation_origin, plane)
        ref_dir_vec = get_reflected_vector(self.rotation_axis, plane)
        ref_dir_vec = -ref_dir_vec
        new_edit = self.__class__(operand=operand, rotation_axis=ref_dir_vec,
                                  rotation_origin=ref_point_vec, restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        # Rotation of things in a rotation symmetry is tricky.
        rotated_pt_vec = get_rotated_point(
            self.rotation_origin, origin, rot_vec, angle)
        rotated_dir_vec = get_rotated_vector(
            self.rotation_axis, rot_vec, angle)
        new_edit = self.__class__(operand=operand, rotation_axis=rotated_dir_vec,
                                  rotation_origin=rotated_pt_vec, restrictor_name=rot_name, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        shifted_origin = get_translated_point(self.rotation_origin, delta)
        new_edit = self.__class__(operand=operand, rotation_axis=self.rotation_axis,
                                  rotation_origin=shifted_origin, restrictor_name=trans_name, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                if amount_equal:
                    line_a = Line(point=self.rotation_origin,
                                  direction=self.rotation_axis)
                    line_b = Line(point=edit.rotation_origin,
                                  direction=edit.rotation_axis)
                    valid = line_a == line_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(
                    self.amount - edit.amount, mode=mode, value=multiplier)
                if amount_equal:
                    line_a = Line(point=self.rotation_origin,
                                  direction=self.rotation_axis)
                    line_b = Line(point=edit.rotation_origin,
                                  direction=edit.rotation_axis)
                    valid = line_a == line_b
        return valid

    def signature(self, shape):
        origin_string = get_point_string(
            self.rotation_origin, self.operand, shape)
        sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if sign == 1:
            amount_str = get_amount_str(self.amount)
            direction_str = get_direction_string(
                self.rotation_axis, self.operand)
        else:
            amount_str = get_amount_str(-self.amount)
            direction_str = get_direction_string(
                -self.rotation_axis, self.operand)
        sig = f"Rotate the {self.restrictor_name} {self.restrictor_type} of {self.operand.full_label} by {amount_str} with rotation axis pointing towards its {direction_str} and rotation origin at {origin_string}."
        return sig


class PointRotate(RestrictedRotate):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, rotation_origin: sp.Matrix, rotation_axis: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(PointRotate, self).__init__(operand=operand, rotation_origin=rotation_origin,
                                          rotation_axis=rotation_axis, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeRotate(RestrictedRotate):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, rotation_origin: sp.Matrix, rotation_axis: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(EdgeRotate, self).__init__(operand=operand, rotation_origin=rotation_origin,
                                         rotation_axis=rotation_axis, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceRotate(RestrictedRotate):
    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, rotation_origin: sp.Matrix, rotation_axis: sp.Matrix, restrictor_name, amount=None, identifier=None):
        super(FaceRotate, self).__init__(operand=operand, rotation_origin=rotation_origin,
                                         rotation_axis=rotation_axis, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class PartScale1D(PartEdit):
    def __init__(self, operand, origin, direction, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.origin = sp.Matrix(1, 3, origin)
        self.direction = sp.Matrix(1, 3, direction).normalized()
        self.param_names = ['direction', 'origin']

    def apply(self, geometric_expr):
        # scale by self.amount
        geometric_expr = part_scale_1d(geometric_expr, self.origin,
                                       self.direction, 1 + self.amount)

        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_dir_vec = get_reflected_vector(self.direction, plane)
        ref_point_vec = get_reflected_point(self.origin, plane)
        new_edit = PartScale1D(operand=operand, direction=ref_dir_vec, origin=ref_point_vec,
                               amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        rotated_dir_vec = get_rotated_vector(self.direction, rot_vec, angle)
        rotated_pt_vec = get_rotated_point(self.origin, origin, rot_vec, angle)
        new_edit = PartScale1D(operand=operand, direction=rotated_dir_vec, origin=rotated_pt_vec,
                               amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = PartScale1D(operand=operand, direction=self.direction, origin=shifted_origin,
                               amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartScale1D):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            if amount_equal:
                plane_a = Plane(point=self.origin, normal=self.direction)
                plane_b = Plane(point=edit.origin, normal=edit.direction)
                valid = plane_a == plane_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartScale1D):
            amount_equal = evaluate_equals_zero(
                self.amount - edit.amount, mode=mode, value=multiplier)
            if amount_equal:
                plane_a = Plane(point=self.origin, normal=self.direction)
                plane_b = Plane(point=edit.origin, normal=edit.direction)
                valid = plane_a == plane_b
        return valid

    def signature(self, shape):
        direction = get_direction_string(self.direction, self.operand)
        opp_dir = get_direction_string(-self.direction, self.operand)
        bidirection_string = get_bidirection_string(
            self.direction, self.operand)
        origin_string = get_point_string(self.origin, self.operand, shape)
        # Expand or contract
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(self.amount)
            explaination_str = f"{direction} face further {direction} and {opp_dir} face further {opp_dir}"
            dir_str = direction
        else:
            sign_str = "Contract"
            amount_str = get_amount_str(-self.amount)
            explaination_str = f"{direction} face {opp_dir} and {opp_dir} face {direction}"
            dir_str = opp_dir
        # Not always correct, esp when the origin is outside the object.
        # check if the origin is within the object
        if not '-' in direction:
            face_vec_1 = self.operand.face_center(direction) - self.origin
            face_vec_2 = self.operand.face_center(opp_dir) - self.origin
            face_dot_1 = face_vec_1.dot(self.direction)
            face_dot_2 = face_vec_2.dot(self.direction)
            if sp.sign(face_dot_1) == sp.sign(face_dot_2):
                sig = f"{sign_str} the {self.operand.full_label} by {amount_str} by shifting its {bidirection_string} faces further to the {dir_str} direction with the scaling origin at {origin_string}."
            else:
                sig = f"{sign_str} the {self.operand.full_label} by {amount_str} by moving its {explaination_str}. The scaling origin is at {origin_string}."
        else:
            sig = f"{sign_str} the {self.operand.full_label} by {amount_str} by shifting its {bidirection_string} faces further to the {dir_str} direction with the scaling origin at {origin_string}."
        return sig


class RestrictedScale1D(RestrictedEdit):
    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.origin = sp.Matrix(1, 3, origin)
        self.direction = sp.Matrix(1, 3, direction).normalized()
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ['direction', 'origin', 'restrictor_name']

        self.unrestricted_class = PartScale1D

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        scale_list = []
        for i in range(n_points):
            if i in valid_indices:
                point_vec = geometric_expr[i, :]
                point_vec = point_vec - self.origin
                dot_prod = point_vec.dot(self.direction)
                along_dir_proj = self.direction * dot_prod
                other_dir_proj = point_vec - along_dir_proj
                point_vec = (along_dir_proj * (1 + self.amount)) + \
                    other_dir_proj
                point_vec = point_vec + self.origin
                scale_list.append(point_vec)
            else:
                scale_list.append(geometric_expr[i, :])
        scale_stacked = sp.Matrix.vstack(*scale_list)
        return scale_stacked

    def get_reflected(self, operand, plane, ref_name):
        ref_dir_vec = get_reflected_vector(self.direction, plane)
        ref_point_vec = get_reflected_point(self.origin, plane)
        new_edit = self.__class__(operand=operand, direction=ref_dir_vec, origin=ref_point_vec,
                                  restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        rotated_dir_vec = get_rotated_vector(self.direction, rot_vec, angle)
        rotated_pt_vec = get_rotated_point(self.origin, origin, rot_vec, angle)
        new_edit = self.__class__(operand=operand, direction=rotated_dir_vec,
                                  origin=rotated_pt_vec, restrictor_name=rot_name,
                                  amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = self.__class__(operand=operand, direction=self.direction, origin=shifted_origin,
                                  restrictor_name=trans_name, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                if amount_equal:
                    plane_a = Plane(point=self.origin, normal=self.direction)
                    plane_b = Plane(point=edit.origin, normal=edit.direction)
                    valid = plane_a == plane_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(
                    self.amount - edit.amount, mode=mode, value=multiplier)
                if amount_equal:
                    plane_a = Plane(point=self.origin, normal=self.direction)
                    plane_b = Plane(point=edit.origin, normal=edit.direction)
                    valid = plane_a == plane_b
        return valid

    def signature(self, shape):
        bidirection_string = get_bidirection_string(
            self.direction, self.operand)
        origin_string = get_point_string(self.origin, self.operand, shape)
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(self.amount)
        else:
            sign_str = "Contract"
            amount_str = get_amount_str(-self.amount)
        sig = f"{sign_str} the {self.restrictor_name} {self.restrictor_type} of {self.operand.full_label} by {amount_str} along its {bidirection_string} direction with the scaling origin at {origin_string}."
        return sig


class PointScale1D(RestrictedScale1D):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(PointScale1D, self).__init__(operand=operand, origin=origin, direction=direction,
                                           restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeScale1D(RestrictedScale1D):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(EdgeScale1D, self).__init__(operand=operand, origin=origin, direction=direction,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceScale1D(RestrictedScale1D):

    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(FaceScale1D, self).__init__(operand=operand, origin=origin, direction=direction,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class PartScale2D(PartEdit):

    def __init__(self, operand, origin, plane_normal, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.plane_normal = sp.Matrix(1, 3, plane_normal)
        self.origin = sp.Matrix(1, 3, origin)
        self.param_names = ['plane_normal', 'origin']

    def apply(self, geometric_expr):
        # scale by self.amount

        center = self.origin
        center_stack = sp.Matrix.vstack(
            *[center,]*geometric_expr.shape[0])
        geometric_expr = geometric_expr - center_stack

        stack_normal = sp.Matrix.vstack(
            *[self.plane_normal,]*geometric_expr.shape[0])
        dot_prod_stacked = geometric_expr.multiply_elementwise(
            stack_normal)
        dot_prod = dot_prod_stacked * sp.ones(dot_prod_stacked.shape[1], 1)
        scaled_normal = stack_normal.multiply_elementwise(
            sp.Matrix.hstack(dot_prod, dot_prod, dot_prod))
        projection = geometric_expr - scaled_normal
        geometric_expr = (projection * (1 + self.amount)) + scaled_normal

        geometric_expr = geometric_expr + center_stack

        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_point = get_reflected_point(self.origin, plane)
        ref_plane_normal = get_reflected_vector(self.plane_normal, plane)
        new_edit = PartScale2D(
            operand=operand, plane_normal=ref_plane_normal, origin=ref_point, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        rot_point = get_rotated_point(self.origin, origin, rot_vec, angle)
        rot_plane_normal = get_rotated_vector(
            self.plane_normal, rot_vec, angle)
        new_edit = PartScale2D(
            operand=operand, plane_normal=rot_plane_normal, origin=rot_point, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = PartScale2D(operand=operand, plane_normal=self.plane_normal,
                               origin=shifted_origin, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartScale2D):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            if amount_equal:
                line_a = Line(point=self.origin, direction=self.plane_normal)
                line_b = Line(point=edit.origin, direction=edit.plane_normal)
                valid = line_a == line_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartScale2D):
            amount_equal = evaluate_equals_zero(
                self.amount - edit.amount, mode=mode, value=multiplier)
            if amount_equal:
                line_a = Line(point=self.origin, direction=self.plane_normal)
                line_b = Line(point=edit.origin, direction=edit.plane_normal)
                valid = line_a == line_b
        return valid

    def signature(self, shape):
        origin_string = get_point_string(self.origin, self.operand, shape)
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(self.amount)
            move_str = "away from"
        else:
            sign_str = "Contract"
            move_str = "towards"
            amount_str = get_amount_str(-self.amount)
        normal_dir = get_direction_string(self.plane_normal, self.operand)
        opposite_dir = get_direction_string(-self.plane_normal, self.operand)
        if "-" not in normal_dir:
            scaled_faces = [x for x in DIR_NAME if x !=
                            normal_dir and x != opposite_dir]
            scaled_faces_str = ', '.join(scaled_faces) + " faces"
            sig = f"{sign_str} the {self.operand.full_label} by moving the {scaled_faces_str} {move_str} the {origin_string} by {amount_str}."
        else:
            sig = f"{sign_str} the {self.operand.full_label} along the plane at {origin_string} with the plane normal pointing towards {self.operand.full_label}'s {normal_dir} by {amount_str}."
        return sig


class RestrictedScale2D(RestrictedEdit):

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, plane_normal: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.plane_normal = sp.Matrix(1, 3, plane_normal)
        self.origin = sp.Matrix(1, 3, origin)
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ['plane_normal', 'origin', 'restrictor_name']

        self.unrestricted_class = PartScale2D

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        scale_list = []
        for i in range(n_points):
            if i in valid_indices:
                point_vec = geometric_expr[i, :]
                point_vec = point_vec - self.origin
                dot_prod = point_vec.dot(self.plane_normal)
                scaled_normal = self.plane_normal * dot_prod
                projection = point_vec - scaled_normal
                point_vec = (projection * (1 + self.amount)) + scaled_normal
                point_vec = point_vec + self.origin
                scale_list.append(point_vec)
            else:
                scale_list.append(geometric_expr[i, :])
        scale_stacked = sp.Matrix.vstack(*scale_list)
        return scale_stacked

    def get_reflected(self, operand, plane, ref_name):
        ref_point = get_reflected_point(self.origin, plane)
        ref_plane_normal = get_reflected_vector(self.plane_normal, plane)
        new_edit = self.__class__(operand=operand, plane_normal=ref_plane_normal, origin=ref_point,
                                  restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        rot_point = get_rotated_point(self.origin, origin, rot_vec, angle)
        rot_plane_normal = get_rotated_vector(
            self.plane_normal, rot_vec, angle)
        new_edit = self.__class__(operand=operand, plane_normal=rot_plane_normal, origin=rot_point,
                                  restrictor_name=rot_name, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = self.__class__(operand=operand, plane_normal=self.plane_normal,
                                  restrictor_name=trans_name, origin=shifted_origin, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                if amount_equal:
                    line_a = Line(point=self.origin,
                                  direction=self.plane_normal)
                    line_b = Line(point=edit.origin,
                                  direction=edit.plane_normal)
                    valid = line_a == line_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(
                    self.amount - edit.amount, mode=mode, value=multiplier)
                if amount_equal:
                    line_a = Line(point=self.origin,
                                  direction=self.plane_normal)
                    line_b = Line(point=edit.origin,
                                  direction=edit.plane_normal)
                    valid = line_a == line_b
        return valid

    def signature(self, shape):
        origin_string = get_point_string(self.origin, self.operand, shape)
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(self.amount)
            scale_str = "expanding"
        else:
            sign_str = "Contract"
            scale_str = "contracting"
            amount_str = get_amount_str(self.amount)
        normal_dir = get_direction_string(self.plane_normal, self.operand)
        sig = f"{sign_str} the {self.restrictor_name} {self.restrictor_type} of {self.operand.full_label} along the plane at {origin_string} with the plane normal pointing towards {self.operand.full_label}'s {normal_dir} by {amount_str}."
        return sig


class PointScale2D(RestrictedScale2D):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, plane_normal: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(PointScale2D, self).__init__(operand=operand, origin=origin, plane_normal=plane_normal,
                                           restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeScale2D(RestrictedScale2D):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, plane_normal: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(EdgeScale2D, self).__init__(operand=operand, origin=origin, plane_normal=plane_normal,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceScale2D(RestrictedScale2D):
    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, plane_normal: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(FaceScale2D, self).__init__(operand=operand, origin=origin, plane_normal=plane_normal,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class PartScale3D(PartEdit):
    def __init__(self, operand, origin, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.origin = sp.Matrix(1, 3, origin)
        self.param_names = ["origin"]

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        # scale by self.amount
        origin_stack = sp.Matrix.vstack(
            *[self.origin,]*geometric_expr.shape[0])
        geometric_expr = geometric_expr - origin_stack
        geometric_expr = geometric_expr * (1 + self.amount)
        geometric_expr = geometric_expr + origin_stack

        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_point_vec = get_reflected_point(self.origin, plane)
        new_edit = PartScale3D(
            operand=operand, origin=ref_point_vec, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        rotated_pt_vec = get_rotated_point(self.origin, origin, rot_vec, angle)
        new_edit = PartScale3D(
            operand=operand, origin=rotated_pt_vec, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = PartScale3D(
            operand=operand, origin=shifted_origin, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartScale3D):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            if amount_equal:
                point_equal = (
                    self.origin - edit.origin).norm() < COMPARISON_THRESHOLD
                valid = point_equal
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartScale3D):
            amount_equal = evaluate_equals_zero(
                self.amount - edit.amount, mode=mode, value=multiplier)
            if amount_equal:
                point_equal = (
                    self.origin - edit.origin).norm() < COMPARISON_THRESHOLD
                valid = point_equal
        return valid

    def signature(self, shape):
        origin_string = get_point_string(self.origin, self.operand, shape)
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(self.amount)
        else:
            sign_str = "Contract"
            amount_str = get_amount_str(-self.amount)
        sig = f"{sign_str} the {self.operand.full_label} by {amount_str} in all the directions with the scaling origin at {origin_string}."
        return sig


class RestrictedScale3D(RestrictedEdit):

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.origin = sp.Matrix(1, 3, origin)
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ["origin", "restrictor_name"]

        self.unrestricted_class = PartScale3D

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        scale_list = []
        for i in range(n_points):
            if i in valid_indices:
                point_vec = geometric_expr[i, :]
                point_vec = point_vec - self.origin
                point_vec = point_vec * (1 + self.amount)
                point_vec = point_vec + self.origin
                scale_list.append(point_vec)
            else:
                scale_list.append(geometric_expr[i, :])
        scale_stacked = sp.Matrix.vstack(*scale_list)
        return scale_stacked

    def get_reflected(self, operand, plane, ref_name):
        ref_point_vec = get_reflected_point(self.origin, plane)
        new_edit = self.__class__(operand=operand, origin=ref_point_vec,
                                  restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        rotated_pt_vec = get_rotated_point(self.origin, origin, rot_vec, angle)
        new_edit = self.__class__(operand=operand, origin=rotated_pt_vec,
                                  restrictor_name=rot_name, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        shifted_origin = get_translated_point(self.origin, delta)
        new_edit = self.__class__(
            operand=operand, origin=shifted_origin, restrictor_name=trans_name, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                if amount_equal:
                    point_equal = (
                        self.origin - edit.origin).norm() < COMPARISON_THRESHOLD
                    valid = point_equal
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(
                    self.amount - edit.amount, mode=mode, value=multiplier)
                if amount_equal:
                    point_equal = (
                        self.origin - edit.origin).norm() < COMPARISON_THRESHOLD
                    valid = point_equal
        return valid

    def signature(self, shape):
        origin_string = get_point_string(self.origin, self.operand, shape)
        amount_sign = sp.sign(self.amount.coeff(MAIN_VAR))
        if amount_sign == 1:
            sign_str = "Expand"
            amount_str = get_amount_str(-self.amount)
        else:
            sign_str = "Contract"
            amount_str = get_amount_str(-self.amount)
        sig = f"{sign_str} the {self.restrictor_name} {self.restrictor_type} of {self.operand.full_label} by {amount_str} with the scaling origin at {origin_string}."
        return sig


class PointScale3D(RestrictedScale3D):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(PointScale3D, self).__init__(operand=operand, origin=origin,
                                           restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeScale3D(RestrictedScale3D):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(EdgeScale3D, self).__init__(operand=operand, origin=origin,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceScale3D(RestrictedScale3D):
    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, origin: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(FaceScale3D, self).__init__(operand=operand, origin=origin,
                                          restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class PartShear(PartEdit):
    def __init__(self, operand, shear_plane_origin, shear_plane_normal, shear_direction, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.shear_direction = sp.Matrix(1, 3, shear_direction).normalized()
        self.shear_plane_origin = sp.Matrix(1, 3, shear_plane_origin)
        self.shear_plane_normal = sp.Matrix(
            1, 3, shear_plane_normal).normalized()
        self.param_names = ['shear_direction',
                            'shear_plane_origin', 'shear_plane_normal']

    def apply(self, geometric_expr):
        # scale by self.amount

        shear_matrix = sp.eye(3)
        for i in range(3):
            for j in range(3):
                # Update the shear matrix components based on the direction and plane_normal
                shear_matrix[i, j] += self.amount * \
                    self.shear_direction[i] * self.shear_plane_normal[j]
        shift_stack = sp.Matrix.vstack(
            *[self.shear_plane_origin,]*geometric_expr.shape[0])
        geometric_expr = geometric_expr - shift_stack
        geometric_expr = geometric_expr * shear_matrix.T
        geometric_expr = geometric_expr + shift_stack
        return geometric_expr

    def get_reflected(self, operand, plane):
        ref_point_vec = get_reflected_point(self.shear_plane_origin, plane)
        ref_plane_dir_vec = get_reflected_vector(
            self.shear_plane_normal, plane)
        reflected_vec = get_reflected_vector(self.shear_direction, plane)
        new_edit = PartShear(operand=operand, shear_direction=reflected_vec, shear_plane_origin=ref_point_vec,
                             shear_plane_normal=ref_plane_dir_vec, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        # Rotation of things in a rotation symmetry is tricky.
        rotated_pt_vec = get_rotated_point(
            self.shear_plane_origin, origin, rot_vec, angle)
        rotated_plane_dir_vec = get_rotated_vector(
            self.shear_plane_normal, rot_vec, angle)
        shear_dir_vec = get_rotated_vector(
            self.shear_direction, rot_vec, angle)
        new_edit = PartShear(operand=operand, shear_direction=shear_dir_vec, shear_plane_origin=rotated_pt_vec,
                             shear_plane_normal=rotated_plane_dir_vec, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta):
        shifted_origin = get_translated_point(self.shear_plane_origin, delta)
        new_edit = PartShear(operand=operand, shear_direction=self.shear_direction, shear_plane_origin=shifted_origin,
                             shear_plane_normal=self.shear_plane_normal, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, PartShear):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            if amount_equal:
                var_dir_equal = self.shear_direction.cross(
                    edit.shear_direction).norm() < COMPARISON_THRESHOLD
                if var_dir_equal:
                    plane_a = Plane(point=self.shear_plane_origin,
                                    normal=self.shear_plane_normal)
                    plane_b = Plane(point=edit.shear_plane_origin,
                                    normal=edit.shear_plane_normal)
                    valid = plane_a == plane_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, PartShear):
            amount_equal = evaluate_equals_zero(
                self.amount - edit.amount, mode=mode, value=multiplier)
            if amount_equal:
                var_dir_equal = self.shear_direction.cross(
                    edit.shear_direction).norm() < COMPARISON_THRESHOLD
                if var_dir_equal:
                    plane_a = Plane(point=self.shear_plane_origin,
                                    normal=self.shear_plane_normal)
                    plane_b = Plane(point=edit.shear_plane_origin,
                                    normal=edit.shear_plane_normal)
                    valid = plane_a == plane_b
        return valid

    def signature(self, shape):
        direction_string = get_direction_string(
            self.shear_direction, self.operand)
        shear_center = get_point_string(
            self.shear_plane_origin, self.operand, shape)
        dir_str = get_direction_string(self.shear_plane_origin, self.operand)
        opp_direction = get_direction_string(
            -self.shear_plane_origin, self.operand)
        sig = f"Shear the {self.operand.full_label} by {get_amount_str(self.amount)} along the {direction_string} with the shearing center at {shear_center}. Shearing affects both {dir_str} and {opp_direction} face if the shearing center is at the part's center."
        return sig


class RestrictedShear(RestrictedEdit):
    def __init__(self, operand: OP_TYPE, shear_plane_origin: sp.Matrix, shear_plane_normal: sp.Matrix, shear_direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        self.shear_direction = sp.Matrix(1, 3, shear_direction).normalized()
        self.shear_plane_origin = sp.Matrix(1, 3, shear_plane_origin)
        self.shear_plane_normal = sp.Matrix(
            1, 3, shear_plane_normal).normalized()
        self.restrictor_name = self.get_restrictor_name(restrictor_name)
        self.param_names = ['shear_direction', 'shear_plane_origin',
                            'shear_plane_normal', 'restrictor_name']

        self.unrestricted_class = PartShear

    def apply(self, geometric_expr):
        # geometric_expr is 3 x N sp.Matrix
        shear_matrix = sp.eye(3)
        for i in range(3):
            for j in range(3):
                # Update the shear matrix components based on the direction and plane_normal
                shear_matrix[i, j] += self.amount * \
                    self.shear_direction[i] * self.shear_plane_normal[j]
        n_points = geometric_expr.shape[0]
        # only apply on the corresponding edge
        valid_indices = self.operand.name_to_indices[self.restrictor_name]
        shear_list = []
        for i in range(n_points):
            if i in valid_indices:
                point_vec = geometric_expr[i, :]
                point_vec = point_vec - self.shear_plane_origin
                point_vec = point_vec * shear_matrix.T
                point_vec = point_vec + self.shear_plane_origin
                shear_list.append(point_vec)
            else:
                shear_list.append(geometric_expr[i, :])
        shear_stacked = sp.Matrix.vstack(*shear_list)
        return shear_stacked

    def get_reflected(self, operand, plane, ref_name):
        ref_point_vec = get_reflected_point(self.shear_plane_origin, plane)
        ref_plane_dir_vec = get_reflected_vector(
            self.shear_plane_normal, plane)
        reflected_vec = get_reflected_vector(self.shear_direction, plane)
        new_edit = self.__class__(operand=operand, shear_direction=reflected_vec, shear_plane_origin=ref_point_vec,
                                  shear_plane_normal=ref_plane_dir_vec, restrictor_name=ref_name, amount=self.amount)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        # Rotation of things in a rotation symmetry is tricky.
        rotated_pt_vec = get_rotated_point(
            self.shear_plane_origin, origin, rot_vec, angle)
        rotated_plane_dir_vec = get_rotated_vector(
            self.shear_plane_normal, rot_vec, angle)
        shear_dir_vec = get_rotated_vector(
            self.shear_direction, rot_vec, angle)
        new_edit = self.__class__(operand=operand, shear_direction=shear_dir_vec, shear_plane_origin=rotated_pt_vec,
                                  restrictor_name=rot_name, shear_plane_normal=rotated_plane_dir_vec, amount=self.amount)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        shifted_origin = get_translated_point(self.shear_plane_origin, delta)
        new_edit = self.__class__(operand=operand, shear_direction=self.shear_direction, shear_plane_origin=shifted_origin,
                                  restrictor_name=trans_name, shear_plane_normal=self.shear_plane_normal, amount=self.amount)
        return new_edit

    def __eq__(self, edit):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(self.amount - edit.amount)
                if amount_equal:
                    var_dir_equal = self.shear_direction.cross(
                        edit.shear_direction).norm() < COMPARISON_THRESHOLD
                    if var_dir_equal:
                        plane_a = Plane(point=self.shear_plane_origin,
                                        normal=self.shear_plane_normal)
                        plane_b = Plane(point=edit.shear_plane_origin,
                                        normal=edit.shear_plane_normal)
                        valid = plane_a == plane_b
        return valid

    def merge_equal(self, edit, multiplier=4, mode=3):
        valid = False
        if isinstance(edit, self.__class__):
            if sorted(self.restrictor_name) == sorted(edit.restrictor_name):
                amount_equal = evaluate_equals_zero(
                    self.amount - edit.amount, mode=mode, value=multiplier)
                if amount_equal:
                    var_dir_equal = self.shear_direction.cross(
                        edit.shear_direction).norm() < COMPARISON_THRESHOLD
                    if var_dir_equal:
                        plane_a = Plane(point=self.shear_plane_origin,
                                        normal=self.shear_plane_normal)
                        plane_b = Plane(point=edit.shear_plane_origin,
                                        normal=edit.shear_plane_normal)
                        valid = plane_a == plane_b
        return valid

    def signature(self, shape):
        direction_string = get_direction_string(
            self.shear_direction, self.operand)
        shear_center = get_point_string(
            self.shear_plane_origin, self.operand, shape)
        sig = f"Shear the {self.restrictor_name} {self.restrictor_type} of {self.operand.full_label} by {get_amount_str(self.amount)} along the {direction_string} with the shearing center at {shear_center}."
        return sig


class PointShear(RestrictedShear):
    restrictor_type = "corner"

    def __init__(self, operand: OP_TYPE, shear_plane_origin: sp.Matrix, shear_plane_normal: sp.Matrix, shear_direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(PointShear, self).__init__(operand=operand, shear_direction=shear_direction, shear_plane_origin=shear_plane_origin,
                                         shear_plane_normal=shear_plane_normal, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class EdgeShear(RestrictedShear):
    restrictor_type = "edge"

    def __init__(self, operand: OP_TYPE, shear_plane_origin: sp.Matrix, shear_plane_normal: sp.Matrix, shear_direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(EdgeShear, self).__init__(operand=operand, shear_direction=shear_direction, shear_plane_origin=shear_plane_origin,
                                        shear_plane_normal=shear_plane_normal, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class FaceShear(RestrictedShear):
    restrictor_type = "face"

    def __init__(self, operand: OP_TYPE, shear_plane_origin: sp.Matrix, shear_plane_normal: sp.Matrix, shear_direction: sp.Matrix, restrictor_name: Tuple[str], amount=None, identifier=None):
        super(FaceShear, self).__init__(operand=operand, shear_direction=shear_direction, shear_plane_origin=shear_plane_origin,
                                        shear_plane_normal=shear_plane_normal, restrictor_name=restrictor_name, amount=amount, identifier=identifier)


class KeepFixed(PartEdit):
    # TODO: Check this.
    def __init__(self, operand, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)

    def apply(self, geometric_expr):
        # scale by self.amount
        return geometric_expr

    def get_reflected(self, operand, plane):
        new_edit = KeepFixed(operand=operand)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle):
        new_edit = KeepFixed(operand=operand)
        return new_edit

    def get_translated(self, operand, delta):
        new_edit = KeepFixed(operand=operand)
        return new_edit

    def __eq__(self, edit):
        if isinstance(edit, KeepFixed):
            return True
        else:
            return False

    def signature(self, shape):
        sig = f"Keep the {self.operand.full_label} fixed."
        return sig


def part_scale_1d(geometric_expr, center, direction, amount):
    # scale by self.amount
    center_stack = sp.Matrix.vstack(
        *[center,]*geometric_expr.shape[0])
    geometric_expr = geometric_expr - center_stack

    stack_normal = sp.Matrix.vstack(
        *[direction,]*geometric_expr.shape[0])
    dot_prod_stacked = geometric_expr.multiply_elementwise(
        stack_normal)
    dot_prod = dot_prod_stacked * sp.ones(dot_prod_stacked.shape[1], 1)
    along_dir_proj = stack_normal.multiply_elementwise(
        sp.Matrix.hstack(dot_prod, dot_prod, dot_prod))
    other_dir_proj = geometric_expr - along_dir_proj
    geometric_expr = (
        along_dir_proj * amount) + other_dir_proj

    geometric_expr = geometric_expr + center_stack

    return geometric_expr


class RelationEdit(Edit):
    ...

    def propagate(self, ):
        self.operand.edit_sequence.append(self)

# When directly called -> Use Part scale on parent, and derive the t-sym td with new_params
# This is because sympy has a hard time solving equations with floor functions.
# With a better solver, we could possibly do it directly as well.


def sym_group_wrapper(operand, stretch_annotation,
                      scaling_type="count",
                      amount=None, identifier=None):

    if scaling_type == "count":
        sym_class = ChangeCount
    elif scaling_type == 'delta':
        sym_class = ChangeDelta
    all_edits = []
    from ..shape_system.relations import TranslationSymmetry, RotationSymmetry
    if isinstance(operand, RotationSymmetry):
        edit = sym_class(operand=operand, stretch_annotation=stretch_annotation,
                         relation_update_mode="BU", amount=amount, identifier=identifier)
        all_edits.append(edit)
    else:
        if stretch_annotation == "keep_fixed":
            # just the edit
            edit = sym_class(operand=operand, stretch_annotation=stretch_annotation,
                             relation_update_mode="BU", amount=amount, identifier=identifier)
            all_edits.append(edit)
        else:
            parent_part = operand.parent_part
            init_param = operand.static_expression()
            start_point, center_point, end_point, delta, count = init_param
            # new_param = operand.dynamic_expression()
            # new_start, new_center, new_end, new_delta, new_count = new_param
            direction = delta.normalized()
            if stretch_annotation == "start":
                stretch = start_point
            elif stretch_annotation == "center":
                stretch = center_point
            elif stretch_annotation == "end":
                stretch = end_point
            # gather the face index closest to the stretch point
            face_names = ['up', 'down', 'left', 'right', 'front', 'back']
            face_centers = [parent_part.face_center(
                face) for face in face_names]
            face_distances = [(stretch - center).norm()
                              for center in face_centers]
            closest_face_center = face_centers[np.argmin(face_distances)]

            new_edit = PartScale1D(parent_part, origin=closest_face_center,
                                   direction=direction, amount=amount)
            all_edits.append(new_edit)
            # Now get the options for the operand
            new_edit_2 = sym_class(operand,
                                   stretch_annotation="F",
                                   relation_update_mode="TD")
            all_edits.append(new_edit_2)

    return all_edits


class ChangeCount(RelationEdit):

    def __init__(self, operand, stretch_annotation="center",
                 relation_update_mode="BU", amount=None, identifier=None):
        if amount is None and relation_update_mode == "BU":
            # Integer addition
            amount = MAIN_VAR * 7
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        from ..shape_system.relations import TranslationSymmetry, RotationSymmetry
        if isinstance(operand, RotationSymmetry):
            self.sym_type = "Rotation"
            self.arc_closed = operand.closed_loop
        elif isinstance(operand, TranslationSymmetry):
            self.sym_type = "Translation"
            self.arc_closed = False

        self.stretch_annotation = stretch_annotation
        self.relation_update_mode = relation_update_mode
        self.param_names = ['stretch_annotation', 'relation_update_mode']
    # option 1: Arc length is changed; get new count
    # option 2: Change Count -> Get new arch length (or start and stop param)
    # option 3: Change Count -> Do not change arch length

    def apply(self, relation_expr, dynamic_param):
        # TODO: Change this to work for general sym. group.
        if self.sym_type == "Translation":
            new_params = self.new_param_translation(
                relation_expr, dynamic_param)
        elif self.sym_type == "Rotation":
            new_params = self.new_param_rotation(relation_expr, dynamic_param)
        return new_params

    def new_param_translation(self, relation_expr, dynamic_param):
        start, center, end, delta, count = relation_expr
        if self.relation_update_mode == "TD":
            dynamic_start, dynamic_end = dynamic_param
            dynamic_len = dynamic_end - dynamic_start
            count_val = dynamic_len.norm() / delta.norm()
            new_count = sp.Max(2, sp.floor(count_val) + 1)
            new_delta = dynamic_len / (new_count - 1)
            new_start = dynamic_start
            new_end = dynamic_end
            new_center = (new_start + new_end) / 2
        else:
            # count_val = sp.Max(1, count + self.amount)
            count_val = count + self.amount
            true_length = (end - start) * (count_val - 1) / (count - 1)
            new_count = sp.Max(2, sp.floor(count_val))
            new_delta = true_length / (new_count - 1)

            new_arc_length = (new_count - 1) * new_delta
            if self.stretch_annotation == 'center':
                # This works only for trans...
                new_start = center - new_arc_length / 2
                new_end = center + new_arc_length / 2
                new_center = center
            elif self.stretch_annotation == 'start':
                new_start = start
                new_end = new_start + new_arc_length
                new_center = (new_start + new_end) / 2
            elif self.stretch_annotation == 'end':
                new_end = end
                new_start = new_end - new_arc_length
                new_center = (new_start + new_end) / 2
            elif self.stretch_annotation == 'keep_fixed':
                # when we want to keep all fixed
                new_delta = (end - start) / (new_count - 1)
                new_center = center
                new_start = start
                new_end = end

        new_params = [new_start, new_center, new_end, new_delta, new_count]
        return new_params

    def new_param_rotation(self, relation_expr, dynamic_param):

        start_point, rotation_origin, axis, angle, count = relation_expr

        if self.relation_update_mode == "TD":
            dynamic_start, dynamic_origin = dynamic_param
            radius_vec = dynamic_start - dynamic_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            new_radius = radius_vec.norm()
            radius_vec = start_point - rotation_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            radius = radius_vec.norm()
            arc_size = radius * angle
            count = 2 * np.pi * new_radius / arc_size
            new_count = count  # + self.amount
            new_count = sp.floor(new_count)
            new_angle = 2 * np.pi / new_count

            # keep the arc length fixed -> Reduce angle
            # total_arc_length = 2 * np.pi * new_radius
            # new_angle = total_arc_length / (radius * new_count)
            new_start = dynamic_start
            new_rot_origin = dynamic_origin
            new_axis = axis
        else:
            # count_val = sp.Max(1, count + self.amount)
            radius_vec = start_point - rotation_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            radius = radius_vec.norm()
            total_arc_length = radius * angle*count
            new_count = count + self.amount
            new_count = sp.floor(new_count)
            # keep the arc length fixed -> Reduce angle
            new_angle = total_arc_length / (radius * new_count)

            new_start = start_point
            new_rot_origin = rotation_origin
            new_axis = axis
        new_params = [new_start, new_rot_origin,
                      new_axis, new_angle, new_count]
        return new_params

    def get_reflected(self, operand, plane, ref_name):
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        # Rotation of things in a rotation symmetry is tricky.
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def __eq__(self, edit):
        if isinstance(edit, self.__class__):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            arc_same = self.arc_closed == edit.arc_closed
            sym_same = self.sym_type == edit.sym_type
            stretch_same = self.stretch_annotation == edit.stretch_annotation
            if amount_equal and arc_same and sym_same and stretch_same:
                return True
            else:
                return False
        else:
            return False

    def merge_equal(self, edit):
        raise NotImplementedError

    def signature(self, shape):
        sym_group = self.operand.signature()
        if self.stretch_annotation == "keep_fixed":
            sig = f"Change the number of elements in the {sym_group}, while keeping its size fixed."
        else:
            sig = f"Change the number of elements in the {sym_group}."
        return sig


class ChangeDelta(RelationEdit):
    def __init__(self, operand, stretch_annotation="center",
                 relation_update_mode="BU", amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)
        from ..shape_system.relations import TranslationSymmetry, RotationSymmetry
        if isinstance(operand, RotationSymmetry):
            self.sym_type = "Rotation"
            self.arc_closed = operand.closed_loop
        elif isinstance(operand, TranslationSymmetry):
            self.sym_type = "Translation"
            self.arc_closed = False

        self.stretch_annotation = stretch_annotation
        self.relation_update_mode = relation_update_mode
        self.param_names = ['stretch_annotation', 'relation_update_mode']

    def apply(self, relation_expr, dynamic_param):
        # TODO: Change this to work for general sym. group.
        if self.sym_type == "Translation":
            new_params = self.new_param_translation(
                relation_expr, dynamic_param)
        elif self.sym_type == "Rotation":
            new_params = self.new_param_rotation(relation_expr, dynamic_param)
        return new_params

    def new_param_translation(self, relation_expr, dynamic_param):
        start, center, end, delta, count = relation_expr
        if self.relation_update_mode == "TD":
            dynamic_start, dynamic_end = dynamic_param
            dynamic_len = dynamic_end - dynamic_start
            new_delta = dynamic_len / (count - 1)
            new_start = dynamic_start
            new_end = dynamic_end
            new_center = (new_start + new_end) / 2
        else:
            new_delta = (1 + self.amount) * delta

            new_arc_length = (count - 1) * new_delta
            if self.stretch_annotation == 'center':
                # This works only for trans...
                new_start = center - new_arc_length / 2
                new_end = center + new_arc_length / 2
                new_center = center
            elif self.stretch_annotation == 'start':
                new_start = start
                new_end = new_start + new_arc_length
                new_center = (new_start + new_end) / 2
            elif self.stretch_annotation == 'end':
                new_end = end
                new_start = new_end - new_arc_length
                new_center = (new_start + new_end) / 2
            else:
                # in this case, we want to equally spread the points
                arc_length = (end - start).norm()
                count_val = arc_length / (new_delta.norm() + 1e-6)
                count = sp.Max(2, sp.floor(count_val) + 1)
                new_delta = (end-start) / (count - 1)
                new_center = center
                # new_arc_length = (count - 1) * new_delta# .norm()
                # new_start = center - new_arc_length / 2
                # new_end = center + new_arc_length / 2
                new_start = start
                new_end = end

        new_params = [new_start, new_center, new_end, new_delta, count]
        return new_params

    def new_param_rotation(self, relation_expr, dynamic_param):

        start_point, rotation_origin, axis, angle, count = relation_expr

        if self.relation_update_mode == "TD":
            dynamic_start, dynamic_origin = dynamic_param
            radius_vec = dynamic_start - dynamic_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            new_radius = radius_vec.norm()
            radius_vec = start_point - rotation_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            radius = radius_vec.norm()

            new_count = count
            new_arc_size = (2 * np.pi * new_radius) / new_count
            angle = new_arc_size / radius

            # keep the arc length fixed -> Reduce angle
            # new_angle = total_arc_length / (radius * new_count)
            new_start = dynamic_start
            new_rot_origin = dynamic_origin
            new_axis = axis
            real_angle = angle
        else:
            # count_val = sp.Max(1, count + self.amount)
            radius_vec = start_point - rotation_origin
            radius_vec = radius_vec - radius_vec.dot(axis) * axis
            radius = radius_vec.norm()
            total_arc_length = radius * angle * count
            new_angle = angle + self.amount
            new_count = total_arc_length / new_angle / radius
            new_count = sp.floor(new_count)
            real_angle = total_arc_length / radius / new_count
            # keep the arc length fixed -> Reduce angle

            new_start = start_point
            new_rot_origin = rotation_origin
            new_axis = axis
        new_params = [new_start, new_rot_origin,
                      new_axis, real_angle, new_count]
        return new_params

    def get_reflected(self, operand, plane, ref_name):
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def get_rotated(self, operand, origin, rot_vec, angle, rot_name):
        # Rotation of things in a rotation symmetry is tricky.
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def get_translated(self, operand, delta, trans_name):
        new_edit = self.__class__(operand=operand, arc_closed=self.arc_closed,
                                  sym_type=self.sym_type, stretch_annotation=self.stretch_annotation)
        return new_edit

    def __eq__(self, edit):
        if isinstance(edit, self.__class__):
            amount_equal = evaluate_equals_zero(self.amount - edit.amount)
            arc_same = self.arc_closed == edit.arc_closed
            sym_same = self.sym_type == edit.sym_type
            stretch_same = self.stretch_annotation == edit.stretch_annotation
            if amount_equal and arc_same and sym_same and stretch_same:
                return True
            else:
                return False
        else:
            return False

    def merge_equal(self, edit):
        raise NotImplementedError

    def signature(self, shape):
        sym_group = self.operand.signature()
        if self.stretch_annotation == "keep_fixed":
            sig = f"Chage the distance between the elements in {sym_group}, while keeping its size fixed. Change the number of instances if required."
        else:
            sig = f"Change the distance between elements in {sym_group}."
        return sig


class DummyContactParamUpdate(RelationEdit):
    def __init__(self, operand, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)

    def apply():
        raise NotImplementedError

    def signature(self, shape):
        sym_group = self.operand.signature()
        sig = f"Update the parameters of {sym_group}."
        return sig


class DummyReflectionUpdateEdit(RelationEdit):
    def __init__(self, operand, amount=None, identifier=None):
        super().__init__(operand=operand, amount=amount, identifier=identifier)

    def apply():
        raise NotImplementedError

    def signature(self, shape):
        sym_group = self.operand.signature()
        sig = f"Update the parameters of {sym_group}."
        return sig


class EditGen():
    def __init__(self, edit_class, param_dict, amount=None):
        self.edit_class = edit_class
        self.param_dict = param_dict
        self.amount = amount

    def employ(self, operand, amount=None):
        if amount is None:
            amount = self.amount
        edit = self.edit_class(
            operand=operand, amount=amount, **self.param_dict)
        return edit

    def __repr__(self):
        param_str_list = [f"{x}={y}" for x, y in self.param_dict.items()]
        param_str = ', '.join([x for x in param_str_list])
        if isinstance(self.amount, sp.Expr):
            amount_str = round_expr(self.amount)
        else:
            amount_str = ""
        string = f"EditGen(edit_class={self.edit_class.__name__}, {param_str}, amount={amount_str})"
        return string

    def signature(self, operand):
        # TODO.
        param_str_list = [f"{x}={y}" for x, y in self.param_dict.items()]

        all_params = [f"operand={operand}"] + param_str_list
        param_str = ', '.join([x for x in all_params])
        string = f"{self.edit_class.__name__}({param_str}, amount={round_expr(self.amount)})"
        return string

    def prompt_signature(self, operand):
        signature = self.edit_class._prompt_signature(
            part=operand, **self.param_dict, amount=self.amount)

        return signature

    def full_signature(self, operand):
        return self.signature(operand)


# relations: Ref. Rot, Trans, ContactRelation
# Not required right now.

HIGHER_TYPE_HINTS = {
    'move': [PartTranslate],
    'keep_fixed': [KeepFixed],
    'tilt': [PartRotate, FaceTranslate, PartShear, FaceRotate, FaceScale1D, FaceScale2D],
    'scale': [PartScale1D, FaceTranslate, PartScale2D, PartScale3D,
              EdgeTranslate, FaceScale1D, FaceScale2D, EdgeScale1D,
              EdgeScale2D],
    'change_count': [ChangeCount, PartScale1D, FaceTranslate, PartScale2D, PartScale3D,
                     EdgeTranslate, FaceScale1D, FaceScale2D, EdgeScale1D,
                     EdgeScale2D],
    'change_delta': [ChangeDelta, PartScale1D, FaceTranslate, PartScale2D, PartScale3D,
                     EdgeTranslate, FaceScale1D, FaceScale2D, EdgeScale1D,
                     EdgeScale2D],  # chane
    # older type hints.
    'translate': [PartTranslate],
    'rotate': [PartRotate, FaceTranslate, PartShear, FaceRotate],
    'shear': [PartRotate, FaceTranslate, PartShear, FaceRotate],


    None: [PartTranslate, FaceTranslate, EdgeTranslate, PartRotate, FaceRotate,
           PartScale1D, PartScale2D, PartScale3D, EdgeScale1D,
           EdgeScale2D, FaceScale1D, FaceScale2D, PartShear,
           ChangeCount, ChangeDelta]
}


def load_edits(shape, edit_gens):
    avoid_edits = (DummyContactParamUpdate,
                   DummyReflectionUpdateEdit, KeepFixed)
    selected_edits = []
    for edit_content in edit_gens:
        edit_gen, operand_type, index = edit_content
        if not issubclass(edit_gen.edit_class, avoid_edits):
            selected_edits.append(edit_content)
    edit_gens = selected_edits
    edits = []
    edited_parts = []
    for edit_content in edit_gens:
        edit_gen, operand_type, index = edit_content
        if operand_type == "part":
            operand = [x for x in shape.partset if x.part_index == index]
            if len(operand) > 0:
                operand = operand[0]
            else:
                continue
        else:
            operand = [x for x in shape.all_relations(
            ) if x.relation_index == index][0]
        edit = edit_gen.employ(operand)
        edits.append(edit)
        edited_parts.append(operand)

    for operand in edited_parts:
        if isinstance(operand, Part):
            operand.state[0] = PART_ACTIVE
            if len(operand.sub_parts) > 0:
                edited_children = [
                    len(child.primitive.edit_sequence) > 0 for child in operand.sub_parts]
                if any(edited_children):
                    print(
                        f"Part {operand.label} has edited children. Turned Off")
                    shape.deactivate_parent(operand)
    # update the shape part states
    return edits, edited_parts
