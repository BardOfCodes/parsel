import itertools
from string import Template

import numpy as np
import scipy
import sympy as sp

from .constants import (RELATION_BROKEN, RELATION_STABLE, RELATION_UNCHECKED, 
                        RELATION_RETAINED, RELATION_REJECTED, RELATION_ACTIVE, 
                        RESOLVE_RELATION, NOTHING_TO_DO, DOT_THRESHOLD, 
                        TEST_VALUE, QUANT_COMPARE_THRESHOLD_MODE_1,
                        RELATION_INACTIVE, PART_INACTIVE, PART_ACTIVE)
from .geometry import COMPARISON_THRESHOLD, Line, Plane
from .math import evaluate_equals_zero, rotation_matrix_sympy
from ..edit_dsl.edits import PartEdit, RestrictedEdit, PartScale1D, ChangeCount, ChangeDelta, get_reflected_point
from ..edit_dsl.edits import (PartTranslate, PartRotate, DummyReflectionUpdateEdit, DummyContactParamUpdate, EditGen, MAIN_VAR)
from ..edit_dsl.edits import (PartScale2D, RestrictedScale2D, PartScale3D, RestrictedScale3D, FaceTranslate)
from scipy.optimize import linear_sum_assignment

# Non homo coords
def global_to_local(point, obb):
    return ((point - obb.get_center()) @ obb.R) / (obb.extent/2)


def local_to_cube(local_coords):
    # for 0
    opp_coords = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ])
    opp_coords = opp_coords[None, :]
    volumes = []
    for i in range(8):
        opp_coord = opp_coords[:, i]
        diff = local_coords - opp_coord
        volume = np.prod(np.abs(diff), axis=1)
        volumes.append(volume)
    volumes = np.array(volumes).T
    volumes = volumes / 8.
    return volumes

class Relation:
    
    def extract_info(self):
        info = {
            "type": self.__class__.__name__,
            "parts": [x.label for x in self.primitives],
            "params": self.params
        }
        return info
class PartRelation(Relation): 
    ...
    # These should  just be propagated as primitive relation to all the primitives that a part has.
class PrimitiveRelation(Relation):
    
    def __init__(self, primitives, relation_index):
        self.primitives = primitives
        for item in primitives:
            item._relations.add(self)
        self.param_names = []
        self.edit_sequence = []
        # for avoiding constant re-evaluation.
        self.state = [RELATION_ACTIVE, RELATION_STABLE, RELATION_UNCHECKED, NOTHING_TO_DO]
        self.summary = ""
        self.importance = 1
        self.relation_index = relation_index
        self._relations = set()
        # self.full_label = f"{self.__class__.__name__}_{relation_index}"
    
    @property
    def operands(self):
        return set([x.part for x in self.primitives])
    
    @property
    def full_label(self):
        return f"{self.__class__.__name__}_{self.relation_index}"
    @property
    def params(self):
        return {x:getattr(self, x) for x in self.param_names}
    
    def get_param_str_list(self):
        param_list = []
        for key, values in self.params.items():
            if isinstance(values, sp.Matrix):
                values = [x[0] for x in values.tolist()]
                # only 3 decimal
                for ind, val in enumerate(values):
                    if val.is_number:
                        values[ind] = f"{val:.2f}"
                    else:
                        values[ind] = str(val)
                values = ", ".join(values)
            else:
                values = values.signature()
            param_list.append(f"{key}=({values})")
        return param_list
    
    
    def __repr__(self):
        primitives_str = ", ".join([x.part.full_label for x in self.primitives])
        repr_string = f"{self.__class__.__name__}({primitives_str})"
        return repr_string
    
    def signature(self):
        primitives_str = ", ".join([x.part.full_label for x in self.primitives])
        repr_string = f"{self.__class__.__name__} between {primitives_str}"
        return repr_string
    
    def full_signature(self):
        return self.signature()
    
    def get_corresponding_restrictor(self, source_part, target_part, original_name, key):
        source_indices = source_part.name_to_indices[original_name]
        matching = self.matching[key]
        resp_indices = tuple(sorted([matching[i] for i in source_indices]))
        reflected_name = target_part.indices_to_name[resp_indices]
        return reflected_name
    
    def remove(self):
        raise ValueError("Do we want this?")
    
    def resolvable(self):
        # i.e. I can make edits to other parts and retain this relation.
        raise NotImplementedError
    
    def automatically_resolvable(self):
        # Generally part relations are resolvable.
        raise NotImplementedError
    
    def resolve(self):
        raise NotImplementedError
    
    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        raise NotImplementedError
    
    def automatically_updatable(self):
        # Generally part relations are resolvable.
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def other_part_edited(self, part):
        primitives = [x for x in self.primitives if x.part != part]
        all_edits = []
        for i in primitives:
            n_edits = len(i.edit_sequence)
            all_edits.append(n_edits)
        min_edit = min(all_edits)
        if min_edit > 0:
            other_part_edited = True
        else:
            other_part_edited = False
        return other_part_edited
    
    def get_unedited_parts(self):
        parts = [x.part for x in self.primitives]
        unedited_parts = [x for x in parts if len(x.primitive.edit_sequence) == 0]
        return unedited_parts
    
    @property
    def parts(self):
        return [x.part for x in self.primitives]
    

    
        
class ReflectionSymmetry(PrimitiveRelation):
    def __init__(self, primitives, plane: Plane, relation_index):
        super().__init__(primitives, relation_index)
        self.plane = plane
        self.param_names = ["plane"]
        self.assign_matching()
        self.record_initial_error()

    def record_initial_error(self):
        delta = self.get_discrepancy()
        initial_errors = []
        for i in range(delta.shape[0]):
            delta_vec = delta[i, :]#.norm()
            initial_errors.append(delta_vec)
        # print(initial_errors)
        self.initial_errors = initial_errors
    
    def assign_matching(self):
        part_1_motion = self.primitives[0].static_expression()
        part_2_motion = self.primitives[1].static_expression()
        
        n_points = part_1_motion.shape[0]
        ref_part_1 = self.get_reflected_points(part_1_motion)

        part_1 = np.array(ref_part_1).astype(np.float32)
        part_2 = np.array(part_2_motion).astype(np.float32)
        # reflect and get matching
        distances = scipy.spatial.distance.cdist(part_1, part_2)
        row_ind, matching = linear_sum_assignment(distances)
        matching = list(matching)
        self.matching = {(0, 1): matching}
        inverse_matching = [matching.index(i) for i in range(n_points)]

        self.matching[(1, 0)] = inverse_matching


    def get_reflected_points(self, part_1_motion):
        n_points = part_1_motion.shape[0]
        plane_origin_stacked = sp.Matrix.vstack(*[self.plane.origin,]*n_points)
        plane_normal_stacked = sp.Matrix.vstack(*[self.plane.normal,]*n_points)

        rel_part_1 = part_1_motion - plane_origin_stacked
        rel_part_dot = rel_part_1.multiply_elementwise(plane_normal_stacked)
        dot_prod = rel_part_dot * sp.Matrix.ones(rel_part_dot.shape[1], 1)
        dot_prod_stacked = sp.Matrix.hstack(*[dot_prod,]*3)
        ref_part_1 = rel_part_1 - plane_normal_stacked.multiply_elementwise(2 * dot_prod_stacked)
        ref_part_1 = ref_part_1 + plane_origin_stacked
        return ref_part_1
    
    def broken(self):
        if len(self.edit_sequence) > 0:
            return False
        all_edits = []
        for i in self.primitives:
            n_edits = len(i.edit_sequence)
            all_edits.append(n_edits)
        min_edit = min(all_edits)
        if min_edit > 0:
            mode = 2
        else:
            mode = 1

        delta = self.get_discrepancy()
        
        n_points = delta.shape[0]

        broken = False
        # get mode
        
        
        for j in range(n_points):
            # for i in range(3):
            # delta_norm = delta[j, :].norm()  - self.initial_errors[j]
            delta_norm = delta[j, :] - self.initial_errors[j]#.norm()
            for i in range(3):
                cur_delta = delta_norm[i]
                equals_zero = evaluate_equals_zero(cur_delta, mode=mode, order=3)
                if not equals_zero:
                    broken = True
                    break
        return broken

    def get_discrepancy(self):

        part_1_motion = self.primitives[0].dynamic_expression()
        part_2_motion = self.primitives[1].dynamic_expression()
        # imagine that it is 8 x 3
        # expression reflection should change the order.
        # position
        ref_part_1 = self.get_reflected_points(part_1_motion)
        # now need the reassign
        matching = self.matching[(1, 0)]
        ref_part_1 = sp.Matrix([ref_part_1[i, :] for i in matching])
        delta = ref_part_1 - part_2_motion
        return delta
    
    
    def prompt_signature(self): 
        sig_template = "Reflection symmetry between ($parts_str) with respect to $plane_str"
        parts_str = ", ".join([f"{x.part.label}" for x in self.primitives])
        plane_str = self.plane.prompt_signature()
        sig_replace_dict = {"parts_str": parts_str, "plane_str": plane_str}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def full_signature(self):
        return self.signature()
    
    def resolvable(self):
        # Always resolvable
        part_1_edits = self.primitives[0].edit_sequence
        part_2_edits = self.primitives[1].edit_sequence
        
        if len(part_1_edits) > 0 and len(part_2_edits) > 0:
            resolve = False
        else:
            resolve = True
        return resolve
    
    def automatically_resolvable(self):
        # when resolvable, it is automatically resolvable.
        return True
    
    def resolve(self,):
        # Much too hacky - to be fixed.
        # remove edits on target part?
        part_1_edits = self.primitives[0].edit_sequence
        part_2_edits = self.primitives[1].edit_sequence
        
        if len(part_1_edits) > 0 and len(part_2_edits) > 0:
            raise ValueError(
                "Cannot resolve reflection symmetry when both parts "
                "already have edits."
            )
        if len(part_1_edits) > len(part_2_edits):
            source_part = self.primitives[0].part
            source_edits = part_1_edits
            target_part = self.primitives[1].part
            key = (0, 1)
        else:
            source_part = self.primitives[1].part
            source_edits = part_2_edits
            target_part = self.primitives[0].part
            key = (1, 0)
        
        new_edits = []
        for edit in source_edits:
            if isinstance(edit, RestrictedEdit):
                reflected_name = self.get_corresponding_restrictor(source_part, target_part, edit.restrictor_name, key)
                new_edit = edit.get_reflected(target_part, self.plane, reflected_name)
            elif isinstance(edit, PartEdit):
                new_edit = edit.get_reflected(operand=target_part, plane=self.plane)
            # add for translation
            new_edits.append(new_edit)
        return new_edits
    
    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # Can be updated only if one of them has an edit,
        # and the edit is not scaling. 
        
        part_1_edits = self.primitives[0].edit_sequence
        part_2_edits = self.primitives[1].edit_sequence
        
        if len(part_1_edits) > 0 and len(part_2_edits) > 0:
            update = False
        else:
            if len(part_1_edits) > len(part_2_edits):
                source_edits = part_1_edits
            else:
                source_edits = part_2_edits
            # need to do a high level analysis but for now:
            update = True
            if len(source_edits) > 1:
                update = False
            else:
                edit = source_edits[0]
                if isinstance(edit, PartTranslate):
                    if sp.Abs(edit.direction.dot(self.plane.normal)) < DOT_THRESHOLD:
                        update = False
                elif isinstance(edit, PartRotate):
                    point = edit.rotation_origin
                    direction = edit.rotation_axis
                    point_normal_projection = (point - self.plane.origin).dot(self.plane.normal)
                    if point_normal_projection > COMPARISON_THRESHOLD:
                        update = False
                    else:
                        if direction.dot(self.plane.normal) > (1 - DOT_THRESHOLD):
                            update = False
                else:
                    update = False
        return update
    
    def automatically_updatable(self):
        return True
    
    def update(self):
        edit = DummyReflectionUpdateEdit(self)
        return [edit,]

    def get_all_interesting_points(self, with_names=False):
        plane_center = self.plane.point
        np_point = np.array(plane_center).astype(np.float32)
        if with_names:
            point_list = [np_point]
            name_list = [f"The center of {self.signature()}'s plane."]
            return point_list, name_list
        else:
            point_list = [np_point]
            return point_list
        

class RotationSymmetry(PrimitiveRelation):
    def __init__(self, primitives, axis: Line, angle, parent_part=None, relation_index=None):
        super().__init__(primitives, relation_index)
        self.axis = axis
        self.angle = sp.Float(angle)
        self.param_names = ["axis", "angle"]
        self.parent_part = parent_part
        parent_part.core_relation = self
        self.assign_matching()
        self.record_initial_error()
        self.setup_relation_specs()
        if self.closed_loop:
            self.angle = sp.Float(2 * np.pi / len(self.primitives))
    
    
    def setup_relation_specs(self):
        self.closed_loop = self.eval_loop_closure()
        if self.closed_loop:
            self.angle = sp.Float(2 * np.pi / len(self.primitives))
        # Need to have the initial arc size as a constant + all the member parts.
        self.count = sp.Integer(len(self.primitives))
        obb = self.parent_part.primitive
        center = np.array(obb.center()).astype(np.float32)
        # size = np.array(obb.size).astype(np.float32)
        # axis = np.array(obb.axis).astype(np.float32)
        # change these

        coord_list = []
        init_center = None
        arc_length = 0
        for ind, instance in enumerate(self.primitives):
            points = instance.static_expression()
            points = np.asarray(points).astype(np.float32)
            # local_coord = ((points - center) @ axis) / (size/2)
            local_coord = global_to_local(points, obb.obb)
            cube_coord = local_to_cube(local_coord)
            coord_list.append(sp.Matrix(cube_coord))
            cur_center = points.mean(axis=0)
            if init_center is None:
                init_center = cur_center
            else:
                distance = np.linalg.norm(cur_center - init_center)
                init_center = cur_center
                arc_length += distance
            if ind in [0, len(self.primitives) - 1]:
                point = points.mean(axis=0)
                # local_coord = ((point - center) @ axis.T) / (size/2)
                local_coords = global_to_local(point, obb.obb)
                if ind == 0:
                    self.start_point = local_to_cube(local_coords)
                else:
                    self.end_point = local_to_cube(local_coords) 
        self.children_relative_coords = coord_list
        self.radius = self.eval_radius() 
        
        # see if its a full circle
        
        if self.closed_loop:
            self.arch_length = self.radius
        else:
            self.arch_length = arc_length
        
        first_obj = self.primitives[0]
        points = first_obj.static_expression()
        points = np.asarray(points).astype(np.float32)
        point = points.mean(axis=0)
        # local_coord = ((point - center) @ axis) / (size/2)
        local_coords = global_to_local(point, obb.obb)
        
        self.start_point = local_to_cube(local_coords)

        last_obj = self.primitives[-1]
        points = last_obj.static_expression()
        points = np.asarray(points).astype(np.float32)
        point = points.mean(axis=0)
        # local_coord = ((point - center) @ axis) / (size/2)
        local_coords = global_to_local(point, obb.obb)
        self.end_point = local_to_cube(local_coords)

        # setup ref. axis
        options = self.parent_part.primitive.axis
        rotation_dir = self.axis.direction
        ref_axis_initial = []
        matching_list = []
        for i in range(3):
            option = options[i, :].normalized()
            if sp.Abs(option.dot(rotation_dir)) < 0.9:
            # pretend its a point from origin to the opetions dir.
                # np_option = np.asarray(option).astype(np.float32)
                # cube_coord = local_to_cube(np_option)
                # ref_axis_initial.append(sp.Matrix(cube_coord))
                ref_axis_initial.append(option)
                parent_expression = self.parent_part.primitive.static_expression()
                # get reflected about center and axis
                center = sp.ones(1, parent_expression.shape[0]) * parent_expression / parent_expression.shape[0]
                plane = Plane(center, option)
                ref_points = [get_reflected_point(parent_expression[i,:], plane) for i in range(parent_expression.shape[0])]
                ref_points = sp.Matrix.vstack(*ref_points)
                part_1 = np.array(parent_expression).astype(np.float32)
                part_2 = np.array(ref_points).astype(np.float32)
                # reflect and get matching
                distances = scipy.spatial.distance.cdist(part_1, part_2)
                row_ind, matching = linear_sum_assignment(distances)
                errors = distances[row_ind, matching]
                matching = list(matching)
                matching_list.append(matching)

        self.ref_axis_initial = list(zip(ref_axis_initial, matching_list))

    def eval_radius(self, static=True):
        if static:
            parent_motion = self.parent_part.primitive.static_expression()
        else:
            parent_motion = self.parent_part.primitive.dynamic_expression()
        first_obj = self.children_relative_coords[0]
        points = first_obj * parent_motion
        if static:
            points = np.asarray(points).astype(np.float32)
            point = points.mean(axis=0)
            point = sp.Matrix(1, 3, point)
        else:
            point = sp.ones(1, points.shape[0]) * points / points.shape[0]
        origin = self.axis.point
        direction = self.axis.direction
        vector = point - origin
        vector = (vector - vector.dot(direction) * direction)
        radius = vector.norm()
        return radius

    def eval_loop_closure(self):
        loop_closed = True
        last_instance = self.primitives[-1]
        last_points = last_instance.static_expression()
        first_instance = self.primitives[0]
        first_points = first_instance.static_expression()
        rot_last_points = self.get_rotated_points(last_points, 1)
        last_index = len(self.primitives) - 1
        matching = self.matching[(last_index, 0)]
        rot_last_points = sp.Matrix([rot_last_points[i, :] for i in matching])
        delta = rot_last_points - first_points
        
        n_points = delta.shape[0]
        error_max = max(self.max_initial_children_error * 1.5, QUANT_COMPARE_THRESHOLD_MODE_1)
        error_mult = error_max/QUANT_COMPARE_THRESHOLD_MODE_1

        for j in range(n_points):
            delta_norm = delta[j, :].norm() # This can be quite costly.
            equals_zero = evaluate_equals_zero(delta_norm, mode=3, value=error_mult)
            if not equals_zero:
                loop_closed = False
                break
        return loop_closed
        # arc length, start_point, end_point, children_coords.
        
    def assign_matching(self):
        all_motions = [x.static_expression() for x in self.primitives]
        n_parts = len(self.primitives)
        part_1_motion = all_motions[0]
        n_points = part_1_motion.shape[0]
        self.matching = {}
        for (ind, part_2_motion) in enumerate(all_motions[1:]):
            # rotate part_1_motion by angle around axis, to get part2 motion
            real_index = ind + 1
            rot_part_1 = self.get_rotated_points(part_1_motion, real_index)
            part_1 = np.array(rot_part_1).astype(np.float32)
            part_2 = np.array(part_2_motion).astype(np.float32)
            # reflect and get matching
            distances = scipy.spatial.distance.cdist(part_1, part_2)
            row_ind, matching = linear_sum_assignment(distances)
            self.matching[(0, real_index)] = list(matching)
        
        for i, j in itertools.product(range(n_parts), range(n_parts)):
            if i == j:
                continue
            key = (i, j)
            if key in self.matching.keys():
                continue
            inverse_key = (j, i)
            if inverse_key in self.matching.keys():
                self.matching[key] = [self.matching[inverse_key].index(i) for i in range(n_points)]
                continue
            matching_1 = self.matching[(0, i)]
            inverse_matching_1 = [matching_1.index(i) for i in range(n_points)]
            matching_2 = self.matching[(0, j)]
            new_matching = [matching_2[i] for i in inverse_matching_1]
            self.matching[key] = new_matching

    def record_initial_error(self):
        delta = self.get_children_discrepancy()
        initial_children_errors = []
        for i in range(delta.shape[0]):
            delta_vec = delta[i, :]#.norm()
            initial_children_errors.append(delta_vec)
        self.initial_children_errors = initial_children_errors
        self.max_initial_children_error = max([x.norm() for x in self.initial_children_errors])

    def broken(self):
        all_edits = []
        for i in self.primitives:
            n_edits = len(i.edit_sequence)
            all_edits.append(n_edits)
        min_edit = min(all_edits)
        if min_edit > 0:
            mode = 2
        else:
            mode = 1

        broken = False
        if self.parent_part.state[0] == PART_INACTIVE:
            delta = self.get_children_discrepancy()
            n_points = delta.shape[0]
            for j in range(n_points):
                delta_norm = delta[j, :] - self.initial_children_errors[j] # This can be quite costly.
                for i in range(3):
                    cur_delta = delta_norm[i]
                    equals_zero = evaluate_equals_zero(cur_delta, mode=mode, order=3)
                    if not equals_zero:
                        broken = True
                        break
        else:
            parent_motion = self.parent_part.primitive.dynamic_expression()
            if self.closed_loop:
                child_coords = self.children_relative_coords[0]
                child_coords = child_coords * parent_motion
                child_center = sp.ones(1, child_coords.shape[0]) * child_coords / child_coords.shape[0]
                radius_vec = (child_center - self.axis.point)
                radius_vec = radius_vec - radius_vec.dot(self.axis.direction) * self.axis.direction
                radius = radius_vec.norm()
                td_arc_length = radius
            else:
                td_arc_length = 0
                init_center = None
                for child_coords in self.children_relative_coords:
                    child_coords = child_coords * parent_motion
                    child_center = sp.ones(1, child_coords.shape[0]) * child_coords
                    if init_center is None:
                        init_center = child_center
                    else:
                        distance = (child_center - init_center).norm()
                        init_center = child_center
                        td_arc_length += distance
            delta = td_arc_length - self.arch_length
            equals_zero = evaluate_equals_zero(delta, mode=mode, order=3)
            if not equals_zero:
                broken = True
        # TODO: REMOVE HACK
        if len(self.edit_sequence) > 0:
            broken = False
        return broken

    
    def get_children_discrepancy(self):

        all_motions = [x.dynamic_expression() for x in self.primitives]
        part_1_motion = all_motions[0]
        n_points = part_1_motion.shape[0]
        all_deltas = []
        for (ind, part_2_motion) in enumerate(all_motions[1:]):
            # rotate part_1_motion by angle around axis, to get part2 motion
            real_index = ind + 1
            rot_part_1 = self.get_rotated_points(part_1_motion, real_index)
            matching = self.matching[(0, real_index)]
            rot_part_1 = sp.Matrix([rot_part_1[i, :] for i in matching])
            delta = rot_part_1 - part_2_motion
            all_deltas.append(delta)
        all_deltas = sp.Matrix.vstack(*all_deltas)
        return all_deltas
    
    def get_rotated_points(self, part_1_motion, ind):
        n_points = part_1_motion.shape[0]
        stacked_origin = sp.Matrix.vstack(*[self.axis.point,]* n_points)
        angle = self.angle * ind
        rotation_matrix = rotation_matrix_sympy(self.axis.direction, angle)
        p_motion = part_1_motion - stacked_origin
        rotated_p_motion = p_motion * rotation_matrix.T
        rot_part_1 = rotated_p_motion + stacked_origin
        return rot_part_1
    
    def resolvable(self):
        # if the relation is broken, and the parent is active I can't resolve it.
        if self.parent_part.state[0] == PART_ACTIVE:
            resolve = False
        else:
            # only if one of them has an edit
            all_edits = []
            for i in self.primitives:
                n_edits = len(i.edit_sequence)
                all_edits.append(n_edits)
            max_edit = max(all_edits)
            all_edits.remove(max_edit)
            if max(all_edits) > 0:
                resolve = False
            else:
                resolve = True
        return resolve
    
    def automatically_resolvable(self):
        # lets see.
        return True
    
    def resolve(self):
        part_edits = [x.edit_sequence for x in self.primitives]
        source_ind = np.argmax([len(x) for x in part_edits])
        source_part = self.primitives[source_ind].part
        origin = self.axis.point
        direction = self.axis.direction
        source_edits = part_edits[source_ind]
        n_parts = len(self.primitives)
        new_edits = []
        for i in range(n_parts):
            if i == source_ind:
                continue
            key = (source_ind, i)
            target_part = self.primitives[i].part
            for edit in source_edits:
                rot_angle = (i - source_ind) * self.angle
                if isinstance(edit, RestrictedEdit):
                    rotated_name = self.get_corresponding_restrictor(source_part, target_part, edit.restrictor_name, key)
                    rotated_edit = edit.get_rotated(self.primitives[i].part, origin,  direction, rot_angle, rotated_name)
                elif isinstance(edit, PartEdit):
                    rotated_edit = edit.get_rotated(self.primitives[i].part, origin,  direction, rot_angle)
                new_edits.append(rotated_edit)
        return new_edits
    
    def updatable(self):
        if self.parent_part.state[0] == PART_ACTIVE:
            if self.closed_loop:
                fixable = True
                edits = self.parent_part.primitive.edit_sequence
                for edit in edits:
                    # if not isinstance(edit, (PartScale2D, RestrictedScale2D, PartScale3D, RestrictedScale3D)):
                    if not isinstance(edit, (PartScale2D, PartScale3D, RestrictedScale3D)):
                        fixable = False
                        break
                # parent_motion = self.parent_part.primitive.dynamic_expression()
                # # this should be reflection sym. about the planes through its center, which are parallel to the axis.
                # center = sp.ones(1, parent_motion.shape[0]) * parent_motion / parent_motion.shape[0]
                # for i in range(2):
                #     cur_ref_axis, matching = self.ref_axis_initial[i]
                #     # inverse_matching = [matching.index(i) for i in range(parent_motion.shape[0])]
                #     # cur_ref_axis = cur_axis * parent_motion
                #     # cur_ref_axis = cur_ref_axis.normalized()
                #     plane = Plane(center, cur_ref_axis)
                #     ref_points = [get_reflected_point(parent_motion[i,:], plane) for i in matching]
                #     ref_points = sp.Matrix.vstack(*ref_points)
                #     delta = parent_motion - ref_points
                #     for i in range(delta.shape[0]):
                #         delta_norm = delta[i, :].norm()
                #         equals_zero = evaluate_equals_zero(delta_norm)
                #         if not equals_zero:
                #             fixable = False
                #             break
                #     if not fixable:
                #         break
            else:
                # TODO: Change this later
                fixable = False        
        else:
            fixable = False
        return fixable
    
    def automatically_updatable(self):
        # two options
        return False
    
    def update(self):
        raise NotImplementedError
    
    def gather_update_options(self):
        # set new R and new theta
        stretch = "F"
        new_edit_1 = EditGen(ChangeCount,
                             param_dict = dict(stretch_annotation=stretch, 
                                               relation_update_mode="TD"))
        new_edit_2 = EditGen(ChangeDelta,
                             param_dict = dict(stretch_annotation=stretch, 
                                               relation_update_mode="TD"))
        options = [new_edit_1, new_edit_2]
        return options
    
    def prompt_signature(self): 
        sig_template = "Rotational symmetry between ($parts_str) with respect to $axis_str"
        parts_str = ", ".join([f"'{x.part.label}'" for x in self.primitives])
        axis_str = self.axis.prompt_signature()
        sig_replace_dict = {"parts_str": parts_str, "axis_str": axis_str}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig
    
    def static_expression(self):
        # start point, radius, angle
        parent_motion = self.parent_part.primitive.static_expression()
        start_point = self.start_point * parent_motion
        rotation_origin = self.axis.point
        axis = self.axis.direction
        angle = self.angle

        return [start_point, rotation_origin, axis, angle, self.count]
    
    def dynamic_expression(self):
        static_expr = self.static_expression()
        dynamic_expr = self.parent_part.primitive.dynamic_expression()
        dynamic_start_point = self.start_point * dynamic_expr
        dynamic_center = static_expr[1].copy()
        for edit in self.parent_part.primitive.edit_sequence:
            dynamic_center = edit.apply(dynamic_center)
        
        dynamic_expr = [dynamic_start_point, dynamic_center]
        for motion in self.edit_sequence:
            static_expr = motion.apply(static_expr, dynamic_expr)
        return static_expr
    
    def resolve_to_update_parent(self):
        if self.closed_loop:
            return None
        else:
            raise ValueError("Not implemented yet")
    
    def execute_relation(self, symbol_dict):
        
        if self.parent_part.state[0] == PART_INACTIVE:
            bboxes = []
            for prim in self.primitives:
                bbox = prim.dynamic_expression()
                bbox = bbox.subs(symbol_dict)
                bboxes.append(bbox)
        
        # option 1: parent mesh - apply transform
        else:
            if len(self.edit_sequence) > 0:

                prim = self.primitives[0]
                bbox = prim.dynamic_expression()
                bbox = bbox - sp.Matrix.vstack(*[prim.center(),]*bbox.shape[0])
                bbox = bbox.subs(symbol_dict)
                relation_params = self.dynamic_expression()
                start_point, rotation_origin, axis, angle, count = relation_params
                start_point = start_point.subs(symbol_dict)
                angle = angle.subs(symbol_dict)
                count = count.subs(symbol_dict)
                rotation_origin = rotation_origin.subs(symbol_dict)
                # parent = self.parent_part.primitive
                # start_point = start_point

                bbox = bbox + sp.Matrix.vstack(*[start_point,]*bbox.shape[0])
                bboxes = [bbox]
                stack_origin = sp.Matrix.vstack(*[rotation_origin,]*bbox.shape[0])
                # obb = self.parent_part.primitive
                # center = np.array(obb.center()).astype(np.float32)
                # m = sp.Matrix.vstack(*[center,]*bbox.shape[0])

                for i in range(1, count):
                    tot_angle = angle * i
                    new_rot_matrix = rotation_matrix_sympy(axis, tot_angle)
                    new_bbox = (bbox - stack_origin) * new_rot_matrix.T + stack_origin
                    # new_bbox = new_bbox + m
                    bboxes.append(new_bbox)
                # prim = self.primitives[0]


            else:
                parent_motion = self.parent_part.primitive.dynamic_expression()
                parent_motion = parent_motion.subs(symbol_dict)
                bboxes = [parent_motion,]
        # Then execute the bboxes, get their cube repr, and apply the above 
        # for all bbox, convert to cube and apply
        # option 3: Relation based execution.
        return bboxes
    
    def get_all_interesting_points(self, with_names=False):
        plane_center = self.axis.point
        np_point = np.array(plane_center).astype(np.float32)
        if with_names:
            point_list = [np_point]
            name_list = [f"The origin of {self.signature()}'s axis"]
            return point_list, name_list
        else:
            point_list = [np_point]
            return point_list
    
    def signature(self):
        parent_str = self.parent_part.full_label 
        repr_string = f"{self.__class__.__name__} between the sub-parts of {parent_str}"
        return repr_string
    
class TranslationSymmetry(PrimitiveRelation):
    def __init__(self, parts, delta: sp.Matrix, parent_part=None, relation_index=None):
        super().__init__(parts, relation_index)
        self.delta = sp.Matrix(1, 3, delta)
        self.param_names = ["delta"]
        self.count = sp.Integer(len(parts))
        self.original_arc_length = self.delta.norm() * (self.count - 1)
        self.parent_part = parent_part
        parent_part.core_relation = self
        self.update_hierarchy_org()
        self.assign_matching()
        self.record_initial_error()
    
    def assign_matching(self):
        all_motions = [x.static_expression() for x in self.primitives]
        n_parts = len(self.primitives)
        part_1_motion = all_motions[0]
        n_points = part_1_motion.shape[0]
        self.matching = {}
        for (ind, part_2_motion) in enumerate(all_motions[1:]):
            # rotate part_1_motion by angle around axis, to get part2 motion
            real_index = ind + 1
            rot_part_1 = self.get_translated_points(part_1_motion, real_index)
            part_1 = np.array(rot_part_1).astype(np.float32)
            part_2 = np.array(part_2_motion).astype(np.float32)
            # reflect and get matching
            distances = scipy.spatial.distance.cdist(part_1, part_2)
            row_ind, matching = linear_sum_assignment(distances)
            self.matching[(0, real_index)] =  list(matching)

        for i, j in itertools.product(range(n_parts), range(n_parts)):
            if i == j:
                continue
            key = (i, j)
            if key in self.matching.keys():
                continue
            inverse_key = (j, i)
            if inverse_key in self.matching.keys():
                self.matching[key] = [self.matching[inverse_key].index(i) for i in range(n_points)]
                continue
            matching_1 = self.matching[(0, i)]
            inverse_matching_1 = [matching_1.index(i) for i in range(n_points)]
            matching_2 = self.matching[(0, j)]
            new_matching = [matching_2[i] for i in inverse_matching_1]
            self.matching[key] = new_matching
            
    def record_initial_error(self):
        delta = self.get_children_discrepancy()
        initial_children_errors = []
        for i in range(delta.shape[0]):
            delta_vec = delta[i, :]#.norm()
            initial_children_errors.append(delta_vec)
        self.initial_children_errors = initial_children_errors
        # for parent do we need to?
        # No since its derived from the relation parameters itself.

    def update_hierarchy_org(self):
        # Get the geo_expr of the primitives
        ...
        obb = self.parent_part.primitive

        first_obj = self.primitives[0]
        points = first_obj.static_expression()
        points = np.asarray(points).astype(np.float32)
        point = points.mean(axis=0)
        # local_coord = ((point - center) @ axis.T) / (size/2)
        local_coord = global_to_local(point, obb.obb)
        self.start_point = local_to_cube(local_coord)

        last_obj = self.primitives[-1]
        points = last_obj.static_expression()
        points = np.asarray(points).astype(np.float32)
        point = points.mean(axis=0)
        # local_coord = ((point - center) @ axis.T) / (size/2)
        local_coord = global_to_local(point, obb.obb)
        self.end_point = local_to_cube(local_coord)
        # in to out?

    def static_expression(self):
        parent_motion = self.parent_part.primitive.static_expression()
        start_point = self.start_point * parent_motion
        end_point = self.end_point * parent_motion
        center_point = (start_point + end_point) / 2
        delta = self.delta
        count = self.count
        relation_param = [start_point, center_point, end_point, delta, count]
        return relation_param
    
    def dynamic_expression(self):
        static_expr = self.static_expression()
        dynamic_expr = self.parent_part.primitive.dynamic_expression()
        dynamic_start_point = self.start_point * dynamic_expr
        dynamic_end_point = self.end_point * dynamic_expr
        dynamic_param = [dynamic_start_point, dynamic_end_point]
        for motion in self.edit_sequence:
            static_expr = motion.apply(static_expr, dynamic_param)
        return static_expr

    def broken(self):
        all_edits = []
        for i in self.primitives:
            n_edits = len(i.edit_sequence)
            all_edits.append(n_edits)
        min_edit = min(all_edits)
        if min_edit > 0:
            mode = 2
        else:
            mode = 1
        if self.parent_part.state[0] == PART_INACTIVE:
            delta = self.get_children_discrepancy()
            
            n_points = delta.shape[0]
            broken = False
            for j in range(n_points):
                delta_norm = delta[j, :] - self.initial_children_errors[j] # This can be quite costly.
                for i in range(3):
                    cur_delta = delta_norm[i]
                    equals_zero = evaluate_equals_zero(cur_delta, mode=mode, order=3)
                    if not equals_zero:
                        broken = True
                        break
        else:
            if len(self.edit_sequence) > 0 and len(self.parent_part.primitive.edit_sequence) > 0:
                broken = False
            else:
                broken = False
                delta = self.get_parent_discrepancy()
                equals_zero = evaluate_equals_zero(delta, mode=mode)
                if not equals_zero:
                    broken = True

        return broken

    def get_translated_points(self, part_1_motion, ind):
        n_points = part_1_motion.shape[0]
        delta_stacked = sp.Matrix.vstack(*[self.delta,]*n_points)
        translated_motion = part_1_motion + delta_stacked * ind
        return translated_motion
    
    
    def get_parent_discrepancy(self):
        
        parent_motion = self.parent_part.primitive.dynamic_expression()
        start_point = self.start_point * parent_motion
        end_point = self.end_point * parent_motion
        td_arc_length = (end_point - start_point).norm()
        
        relation_params = self.dynamic_expression()
        bu_arc_length = (relation_params[2] - relation_params[0]).norm()
        delta = td_arc_length - bu_arc_length
        return delta

    def get_children_discrepancy(self):

        all_motions = [x.dynamic_expression() for x in self.primitives]
        part_1_motion = all_motions[0]
        n_points = part_1_motion.shape[0]
        all_deltas = []
        for (ind, part_2_motion) in enumerate(all_motions[1:]):
            # rotate part_1_motion by angle around axis, to get part2 motion
            real_index = ind + 1
            rot_part_1 = self.get_translated_points(part_1_motion, real_index)
            matching = self.matching[(0, real_index)]
            rot_part_1 = sp.Matrix([rot_part_1[i, :] for i in matching])
            delta = rot_part_1 - part_2_motion
            all_deltas.append(delta)
        # stack them
        all_deltas = sp.Matrix.vstack(*all_deltas)
        return all_deltas

    def resolvable(self):
        if self.parent_part.state[0] == PART_ACTIVE:
            resolve = False
        else:
            all_edits = []
            for i in self.primitives:
                n_edits = len(i.edit_sequence)
                all_edits.append(n_edits)
            max_edit = max(all_edits)
            all_edits.remove(max_edit)
            if max(all_edits) > 0:
                resolve = False
            else:
                resolve = True
        return resolve
    
    def automatically_resolvable(self):
        return True
    
    def resolve(self):
        
        part_edits = [x.edit_sequence for x in self.primitives]
        source_ind = np.argmax([len(x) for x in part_edits])
        source_part = self.primitives[source_ind].part
        delta = self.delta
        source_edits = part_edits[source_ind]
        n_parts = len(self.primitives)
        new_edits = []
        for i in range(n_parts):
            if i == source_ind:
                continue
            key = (source_ind, i)
            target_part = self.primitives[i].part
            for edit in source_edits:
                cur_delta = self.delta * (i - source_ind)
                if isinstance(edit, RestrictedEdit):
                    translated_name = self.get_corresponding_restrictor(source_part, target_part, edit.restrictor_name, key)
                    translated_edit = edit.get_translated(self.primitives[i].part, cur_delta, translated_name)
                elif isinstance(edit, PartEdit):
                    translated_edit = edit.get_translated(self.primitives[i].part, cur_delta)
                new_edits.append(translated_edit)

        return new_edits
    
    def updatable(self):
        if self.parent_part.state[0] == PART_INACTIVE:
            fixable = False
        else:
            fixable = True
            # TODO: Fix this later
            edits = self.parent_part.primitive.edit_sequence
            for edit in edits:
                if isinstance(edit, (PartScale1D, PartScale2D, PartScale3D)):
                    fixable = True
                elif isinstance(edit, FaceTranslate):
                    # Not entirely
                    fixable = True
                else:
                    # Edge Translates 
                    fixable = False
                    break
            # parent_motion = self.parent_part.primitive.dynamic_expression()
            # center_point = sp.one(1, parent_motion.shape[0]) * self.start_point / parent_motion.shape[0]
            # axis = self.delta.normalized()
            # matching = self.parent_reflection_matching
            # cur_ref_axis = axis * parent_motion
            # cur_ref_axis = cur_ref_axis.normalized()
            # plane = Plane(center_point, cur_ref_axis)
            # ref_points = [get_reflected_point(parent_motion[i,:], plane) for i in matching]
            # delta = ref_points - parent_motion
            # for i in range(delta.shape[0]):
            #     delta_norm = delta[i, :].norm()
            #     equals_zero = evaluate_equals_zero(delta_norm)
            #     if not equals_zero:
            #         fixable = False
            #         break
        return fixable
    
    def automatically_updatable(self):
        # This should be true for bottom to up
        if len(self.edit_sequence) > 0:
            return True
        else:
            # need top down fixes
            return False
    
    def update(self):
        # the automatic update
        edit = self.resolve_to_update_parent()
        return [edit,]
        
    def resolve_to_update_parent(self):
        init_param = self.static_expression()
        start_point, center_point, end_point, delta, count = init_param
        new_param = self.dynamic_expression()
        new_start, new_center, new_end, new_delta, new_count = new_param
        
        scale_amount = (new_end - new_start).norm() / (end_point - start_point).norm()
        scale_amount = scale_amount - 1
        direction = new_delta.normalized()
        # start_match = evaluate_equals_zero((new_start - start_point).norm())
        # center_match = evaluate_equals_zero((new_center - center_point).norm())
        # end_match = evaluate_equals_zero((new_end - end_point).norm())

        # TEMP
        X = MAIN_VAR
        start_delta = (new_start.subs({X: TEST_VALUE}) - start_point).norm()
        center_delta = (new_center.subs({X: TEST_VALUE}) - center_point).norm()
        end_delta = (new_end.subs({X: TEST_VALUE}) - end_point).norm()

        min_match = np.argmin([start_delta, center_delta, end_delta])
        if min_match == 0:
            stretch = start_point
        elif min_match == 1:
            stretch = center_point
        else:
            stretch = end_point
        
        new_edit = PartScale1D(self.parent_part, origin=stretch, 
                               direction=direction, amount=scale_amount)

        return new_edit
    
    def gather_update_options(self):
        # set new R and new theta
        stretch = "F"
        new_edit_1 = EditGen(ChangeCount,
                             param_dict = dict(stretch_annotation=stretch, 
                                               relation_update_mode="TD"))
        new_edit_2 = EditGen(ChangeDelta,
                             param_dict = dict(stretch_annotation=stretch, 
                                               relation_update_mode="TD"))
        options = [new_edit_1, new_edit_2]
        return options
    
    
    def prompt_signature(self): 
        sig_template = "Translational symmetry between ($parts_str) with respect to $axis_str"
        parts_str = ", ".join([f"'{x.part.label}'" for x in self.primitives])
        axis_str = self.axis.prompt_signature()
        sig_replace_dict = {"parts_str": parts_str, "axis_str": axis_str}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig
    
    def execute_relation(self, symbol_dict):
        # this is supposed to give the bbox or not variable.
        # option 1: All Children with their transforms.
        if self.parent_part.state[0] == PART_INACTIVE:
            bboxes = []
            for prim in self.primitives:
                bbox = prim.dynamic_expression()
                bbox = bbox.subs(symbol_dict)
                bboxes.append(bbox)
        
        # option 1: parent mesh - apply transform
        else:
            if len(self.edit_sequence) > 0:
                prim = self.primitives[0]
                bbox = prim.dynamic_expression()
                bbox = bbox - sp.Matrix.vstack(*[prim.center(),]*bbox.shape[0])
                bbox = bbox.subs(symbol_dict)
                relation_params = self.dynamic_expression()
                start_point, center_point, end_point, delta, count = relation_params
                count = count.subs(symbol_dict)
                delta = delta.subs(symbol_dict)
                start_point = start_point.subs(symbol_dict)

                bboxes = []
                for i in range(count):
                    position = start_point + i * delta
                    new_bbox = bbox + sp.Matrix.vstack(*[position,]*bbox.shape[0])
                    bboxes.append(new_bbox)
            else:
                parent_motion = self.parent_part.primitive.dynamic_expression()
                parent_motion = parent_motion.subs(symbol_dict)
                bboxes = [parent_motion,]
        # Then execute the bboxes, get their cube repr, and apply the above 
        # for all bbox, convert to cube and apply
        # option 3: Relation based execution.
        return bboxes

    def execute_relation_differentiable(self, symbol_dict):
        # this is supposed to give the bbox or not variable.
        # option 1: All Children with their transforms.
        if self.parent_part.state[0] == PART_INACTIVE:
            bboxes = []
            for prim in self.primitives:
                bbox = prim.dynamic_expression()
                bbox = bbox.subs(symbol_dict)
                bboxes.append(bbox)
        
        # option 1: parent mesh - apply transform
        else:
            if len(self.edit_sequence) > 0:
                prim = self.primitives[0]
                bbox = prim.dynamic_expression()
                bbox = bbox - sp.Matrix.vstack(*[prim.center(),]*bbox.shape[0])

                bbox = bbox#.subs(symbol_dict)
                relation_params = self.dynamic_expression()
                start_point, center_point, end_point, delta, count = relation_params
                count = count.subs(symbol_dict)

                delta = delta.subs(symbol_dict)
                start_point = start_point.subs(symbol_dict)

                bboxes = []
                for i in range(count):
                    position = start_point + i * delta
                    new_bbox = bbox + sp.Matrix.vstack(*[position,]*bbox.shape[0])
                    bboxes.append(new_bbox)
            else:
                parent_motion = self.parent_part.primitive.dynamic_expression()
                parent_motion = parent_motion.subs(symbol_dict)
                bboxes = [parent_motion,]
        # Then execute the bboxes, get their cube repr, and apply the above 
        # for all bbox, convert to cube and apply
        # option 3: Relation based execution.
        return bboxes
    
    def get_all_interesting_points(self, with_names=False):
        point_list = []
        name_list = []
        if with_names:
            return point_list, name_list
        else:
            return point_list
    
    def signature(self):
        parent_str = self.parent_part.full_label 
        repr_string = f"{self.__class__.__name__} between the sub-parts of {parent_str}"
        return repr_string
class FeatureRelation(Relation):
    def __init__(self, features, relation_index):
        self.features = features
        for item in features:
            item._relations.add(self)
        self.state = [RELATION_ACTIVE, RELATION_STABLE, RELATION_UNCHECKED, NOTHING_TO_DO]
        self.summary = ""
        self.importance = 2
        self.param_names = []
        self.edit_sequence = []
        self.relation_index = relation_index
        self.initial_errors = None
        uids = []
        for feature in self.features:
            uids.append(feature.primitive.part.part_index)
        uid = list(set(uids))
        # sort
        uid = sorted(uid)
        self.uid = uid
        self._relations = set()

        self.full_label = f"{self.__class__.__name__}_{relation_index}"
        
    # These should  just be propagated as primitive relation to all the primitives that a part has.
    def extract_info(self):
        info = {
            "type": self.__class__.__name__,
            "parts": [x.label for x in self.primitives.part],
            "params": self.params
        }
        return info
        
    def resolvable(self):
        # i.e. I can make edits to other parts and retain this relation.
        return True
    
    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # Contact relation cannot be precisely defined for update A as A is being edited in a non-affine way.
        return True
    
    @property
    def operands(self):
        return set([x.primitive.part for x in self.features])
    @property
    def params(self):
        return {x:getattr(self, x) for x in self.param_names}
    
    def get_param_str_list(self):
        param_list = []
        for key, values in self.params.items():
            if isinstance(values, sp.Matrix):
                values = [x[0] for x in values.tolist()]
                # only 3 decimal
                for ind, val in enumerate(values):
                    if val.is_number:
                        values[ind] = f"{val:.2f}"
                    else:
                        values[ind] = str(val)
                values = ", ".join(values)
            else:
                values = str(values)
            param_list.append(f"{key}=({values})")
        return param_list
    
    def __repr__(self):
        parts_str = ", ".join([x.full_label for x in self.parts])
        return f"{self.__class__.__name__}({parts_str})"
    
    def signature(self):
        parts_str = ", ".join([x.full_label for x in self.parts])
        return f"Contact Relation between {parts_str}"
    
    def full_signature(self):
        return NotImplementedError()
    def prompt_signature(self):
        raise NotImplementedError()
    
    def remove(self):
        raise ValueError("Do we want this?")
    
    def resolvable(self):
        # Can't fix it if both have edits
        all_edits = []
        for i in self.features:
            n_edits = len(i.primitive.edit_sequence)
            all_edits.append(n_edits)
        min_edits = min(all_edits)
        if min_edits > 0:
            return False
        else:
            return True
    
    def automatically_resolvable(self):
        return False
    
    def resolve(self):
        raise NotImplementedError
    
    def updatable(self):
        # must be implemented in the child class
        raise NotImplementedError
    
    def automatically_updatable(self):
        # Generally part relations are resolvable.
        return True

    def update(self):
        edit = DummyContactParamUpdate(self)
        return [edit,]

    def other_part_edited(self, part):
        primitives = [x.primitive for x in self.features if x.primitive.part != part]
        all_edits = []
        for i in primitives:
            n_edits = len(i.edit_sequence)
            all_edits.append(n_edits)
        min_edit = min(all_edits)
        if min_edit > 0:
            other_part_edited = True
        else:
            other_part_edited = False
        return other_part_edited
    
    def get_unedited_parts(self):
        parts = [x.primitive.part for x in self.features]
        parts = list(set(parts))
        unedited_parts = [x for x in parts if len(x.primitive.edit_sequence) == 0]
        return unedited_parts
    
    @property
    def parts(self):
        parts = [x.primitive.part for x in self.features]
        parts = list(set(parts))
        return parts
    

    def get_all_interesting_points(self, with_names=False):
        point_list = []
        name_list = []
        if with_names:
            return point_list, name_list
        else:
            return point_list
class PointContact(FeatureRelation):
    def __init__(self, feature_group, relation_index):
        super().__init__(feature_group, relation_index)
        broken = self.broken()
        if broken:
            delta = self.get_discrepancy()
            initial_errors = []
            for i in range(delta.shape[0]):
                delta_vec = delta[i, :]#.norm()
                initial_errors.append(delta_vec)
            # print(initial_errors)
            self.initial_errors = initial_errors
        
    def broken(self):
        # we first need the motion vectors for each part.
        # Should we handle approximate GT -> Based on initial discrepancy?
        if len(self.edit_sequence) > 0:
            broken = False
        else:
            broken = False
            all_edits = []
            for i in self.features:
                n_edits = len(i.primitive.edit_sequence)
                all_edits.append(n_edits)
            min_edit = min(all_edits)
            if min_edit > 0:
                mode = 2
            else:
                mode = 1

            delta = self.get_discrepancy()
            for i in range(3):
                delta_i = delta[i]
                if self.initial_errors is not None:
                    delta_i = delta_i - self.initial_errors[i]
                equals_zero = evaluate_equals_zero(delta_i, mode=mode, order=3)
                if not equals_zero:
                    broken = True
                    break
            return broken
        
    def get_discrepancy(self):
        part_1_motion = self.features[0].dynamic_expression()
        part_2_motion = self.features[1].dynamic_expression()
        delta = part_1_motion - part_2_motion
        return delta

    def resolve(self):
        raise NotImplementedError()
    
    
    def prompt_signature(self):
        sig_template = f"Point contact between ($point_str_1, $point_str_2)"
        point_str_1 = self.features[0].prompt_signature()
        point_str_2 = self.features[1].prompt_signature()
        sig_replace_dict = {"point_str_1": point_str_1, "point_str_2": point_str_2}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # there is an edited part, and an unedited part.
        # its updatable only if the dymanic expression of features from edited_part are withint the unedited part.
        # what if it is also being edited?
        all_edits = []
        for i in self.features:
            n_edits = len(i.primitive.edit_sequence)
            all_edits.append(n_edits)
        min_edits = min(all_edits)
        if min_edits > 0:
            # Both have edits
            update = False
        else:
            # if all_edits[0] > all_edits[1]:
            #     edited_part = self.features[0]
            #     unedited_part = self.features[1]
            # else:
            #     edited_part = self.features[1]
            #     unedited_part = self.features[0]
            # X = MAIN_VAR
            # expression = edited_part.dynamic_expression()
            # points = expression.subs({X: TEST_VALUE})
            # # now get the local coord of this.
            # points = np.asarray(points).astype(np.float32)
            # obb = unedited_part.primitive
            # center = np.array(obb.center()).astype(np.float32)
            # size = np.array(obb.size).astype(np.float32)
            # axis = np.array(obb.axis).astype(np.float32)
            # local_coord = ((points - center) @ axis) / (size/2)
            # consider part intersection
            # if np.any(np.abs(local_coord) > 1):
            #     update = False
            # else:
            #     update = True
            update = True # if it depends on numerical then dont

        return update
    
class LineContact(FeatureRelation):
    # 2 point contacts
    def __init__(self, feature_group, relation_index):
        super().__init__(feature_group, relation_index)

        broken = self.broken()
        if broken:
            delta = self.get_discrepancy()
            initial_errors = []
            for i in range(delta.shape[0]):
                delta_vec = delta[i, :]#.norm()
                initial_errors.append(delta_vec)
            # print(initial_errors)
            self.initial_errors = initial_errors
    
    def prompt_signature(self):
        sig_template = f"Line contact between ($point_str_1, $point_str_2)"
        point_str_1 = self.features[0].prompt_signature()
        point_str_2 = self.features[1].prompt_signature()
        sig_replace_dict = {"point_str_1": point_str_1, "point_str_2": point_str_2}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def broken(self):
        if len(self.edit_sequence) > 0:
            broken = False
        else:
            broken = False
            all_edits = []
            for i in self.features:
                n_edits = len(i.primitive.edit_sequence)
                all_edits.append(n_edits)
            min_edit = min(all_edits)
            if min_edit > 0:
                mode = 2
            else:
                mode = 1
            # we first need the motion vectors for each part.
            delta = self.get_discrepancy()
            for i in range(delta.shape[0]):
                delta_i = delta[i, :]
                if self.initial_errors is not None:
                    delta_i = delta_i - self.initial_errors[i]
                for j in range(3):
                    delta_j = delta_i[j]
                    equals_zero = evaluate_equals_zero(delta_j, mode=mode, order=3)
                    if not equals_zero:
                        broken = True
                        break
        return broken
        
    def get_discrepancy(self):

        part_1_motion = self.features[0].dynamic_expression()
        part_2_motion = self.features[1].dynamic_expression()
        part_3_motion = self.features[2].dynamic_expression()
        part_4_motion = self.features[3].dynamic_expression()
        delta_1 = part_1_motion - part_3_motion
        delta_2 = part_2_motion - part_4_motion
        delta = sp.Matrix.vstack(delta_1, delta_2)
        return delta

    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # there is an edited part, and an unedited part.
        # its updatable only if the dymanic expression of features from edited_part are withint the unedited part.
        # what if it is also being edited?
        all_edits = []
        for i in self.features:
            n_edits = len(i.primitive.edit_sequence)
            all_edits.append(n_edits)
        min_edits = min(all_edits)
        if min_edits > 0:
            # Both have edits
            update = False
        else:
            update = True
            # if all_edits[0] > all_edits[2]:
            #     edited_part = self.features[0]
            #     unedited_part = self.features[2]
            # else:
            #     edited_part = self.features[2]
            #     unedited_part = self.features[0]
            # expression = edited_part.dynamic_expression()
            # X = MAIN_VAR
            # points = expression.subs({X: TEST_VALUE})
            # # now get the local coord of this.
            # points = np.asarray(points).astype(np.float32)
            # obb = unedited_part.primitive
            # center = np.array(obb.center()).astype(np.float32)
            # size = np.array(obb.size).astype(np.float32)
            # axis = np.array(obb.axis).astype(np.float32)
            # local_coord = ((points - center) @ axis) / (size/2)
            # if np.any(np.abs(local_coord) > 1):
            #     update = False
            # else:
            #     update = True
        return update
    

class FaceContact(FeatureRelation):
    def __init__(self, feature_group, relation_index):
        super().__init__(feature_group, relation_index)
        broken = self.broken()
        if broken:
            delta = self.get_discrepancy()
            initial_errors = []
            for i in range(delta.shape[0]):
                delta_vec = delta[i, :]#.norm()
                initial_errors.append(delta_vec)
            # print(initial_errors)
            self.initial_errors = initial_errors
        
    def prompt_signature(self):
        sig_template = f"Face contact between ($point_str_1, $point_str_2)"
        point_str_1 = self.features[0].prompt_signature()
        point_str_2 = self.features[1].prompt_signature()
        sig_replace_dict = {"point_str_1": point_str_1, "point_str_2": point_str_2}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def broken(self):
        # we first need the motion vectors for each part.
        if len(self.edit_sequence)> 0:
            broken = False
        else:
            broken = False
            all_edits = []
            for i in self.features:
                n_edits = len(i.primitive.edit_sequence)
                all_edits.append(n_edits)
            min_edit = min(all_edits)
            if min_edit > 0:
                mode = 2
            else:
                mode = 1
            delta = self.get_discrepancy()
            for i in range(delta.shape[0]):
                delta_i = delta[i, :]
                if self.initial_errors is not None:
                    delta_i = delta_i - self.initial_errors[i]
                for j in range(3):
                    delta_j = delta_i[j]
                    equals_zero = evaluate_equals_zero(delta_j, mode=mode, order=3)
                    if not equals_zero:
                        broken = True
                        break
        return broken
        
    def get_discrepancy(self):

        part_1_motion = self.features[0].dynamic_expression()
        part_2_motion = self.features[1].dynamic_expression()
        part_3_motion = self.features[2].dynamic_expression()
        part_4_motion = self.features[3].dynamic_expression()
        part_5_motion = self.features[4].dynamic_expression()
        part_6_motion = self.features[5].dynamic_expression()
        part_7_motion = self.features[6].dynamic_expression()
        part_8_motion = self.features[7].dynamic_expression()
        delta_1 = part_1_motion - part_5_motion
        delta_2 = part_2_motion - part_6_motion
        delta_3 = part_3_motion - part_7_motion
        delta_4 = part_4_motion - part_8_motion
        delta = sp.Matrix.vstack(delta_1, delta_2, delta_3, delta_4)
        return delta


    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # there is an edited part, and an unedited part.
        # its updatable only if the dymanic expression of features from edited_part are withint the unedited part.
        # what if it is also being edited?
        all_edits = []
        for i in self.features:
            n_edits = len(i.primitive.edit_sequence)
            all_edits.append(n_edits)
        min_edits = min(all_edits)
        if min_edits > 0:
            # Both have edits
            update = False
        else:
            # if all_edits[0] > all_edits[4]:
            #     edited_part = self.features[0]
            #     unedited_part = self.features[4]
            # else:
            #     edited_part = self.features[4]
            #     unedited_part = self.features[0]
            # expression = edited_part.dynamic_expression()
            # X = MAIN_VAR
            # points = expression.subs({X: TEST_VALUE})
            # # now get the local coord of this.
            # points = np.asarray(points).astype(np.float32)
            # obb = unedited_part.primitive
            # center = np.array(obb.center()).astype(np.float32)
            # size = np.array(obb.size).astype(np.float32)
            # axis = np.array(obb.axis).astype(np.float32)
            # local_coord = ((points - center) @ axis) / (size/2)
            # if np.any(np.abs(local_coord) > 1):
            #     update = False
            # else:
            #     update = True
            update = True # if it depends on numerical then dont
        return update

class VolumeContact(FeatureRelation):
    
    def __init__(self, feature_group, relation_index):
        super().__init__(feature_group, relation_index)
        broken = self.broken()
        if broken:
            delta = self.get_discrepancy()
            initial_errors = []
            for i in range(delta.shape[0]):
                delta_vec = delta[i, :]#.norm()
                initial_errors.append(delta_vec)
            # print(initial_errors)
            self.initial_errors = initial_errors

    def prompt_signature(self):
        sig_template = f"Volume contact between ($point_str_1, $point_str_2)"
        point_str_1 = self.features[0].prompt_signature()
        point_str_2 = self.features[1].prompt_signature()
        sig_replace_dict = {"point_str_1": point_str_1, "point_str_2": point_str_2}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def broken(self):
        # we first need the motion vectors for each part.
        if len(self.edit_sequence)> 0:
            broken = False
        else:
            broken = False
            all_edits = []
            for i in self.features:
                n_edits = len(i.primitive.edit_sequence)
                all_edits.append(n_edits)
            min_edit = min(all_edits)
            if min_edit > 0:
                mode = 2
            else:
                mode = 1
            delta = self.get_discrepancy()
            for i in range(delta.shape[0]):
                delta_i = delta[i, :]
                if self.initial_errors is not None:
                    delta_i = delta_i - self.initial_errors[i]
                for j in range(3):
                    delta_j = delta_i[j]
                    equals_zero = evaluate_equals_zero(delta_j, mode=mode, order=3)
                    if not equals_zero:
                        broken = True
                        break
        return broken
        
    def get_discrepancy(self):

        part_1_motion = self.features[0].dynamic_expression()
        part_2_motion = self.features[1].dynamic_expression()
        part_3_motion = self.features[2].dynamic_expression()
        part_4_motion = self.features[3].dynamic_expression()
        part_5_motion = self.features[4].dynamic_expression()
        part_6_motion = self.features[5].dynamic_expression()
        part_7_motion = self.features[6].dynamic_expression()
        part_8_motion = self.features[7].dynamic_expression()
        delta_1 = part_1_motion - part_5_motion
        delta_2 = part_2_motion - part_6_motion
        delta_3 = part_3_motion - part_7_motion
        delta_4 = part_4_motion - part_8_motion
        delta = sp.Matrix.vstack(delta_1, delta_2, delta_3, delta_4)
        return delta

    def automatically_resolvable(self):

        part_1_edits = self.features[0].primitive.edit_sequence
        part_2_edits = self.features[4].primitive.edit_sequence
        
        if len(part_1_edits) > len(part_2_edits):
            source_edits = part_1_edits
            target_part = self.features[4].primitive.part
        else:
            source_edits = part_2_edits
            target_part = self.features[0].primitive.part
        auto_resolve = True
        for edit in source_edits:
            if isinstance(edit, RestrictedEdit):
                auto_resolve = False
                break
        return auto_resolve
    
    def resolve(self):
        part_1_edits = self.features[0].primitive.edit_sequence
        part_2_edits = self.features[4].primitive.edit_sequence
        
        if len(part_1_edits) > len(part_2_edits):
            source_edits = part_1_edits
            target_part = self.features[4].primitive.part
        else:
            source_edits = part_2_edits
            target_part = self.features[0].primitive.part
        new_edits = []
        for edit in source_edits:
            copied_edit = edit.copy(operand=target_part)
            new_edits.append(copied_edit)
        return new_edits
    

    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # there is an edited part, and an unedited part.
        # its updatable only if the dymanic expression of features from edited_part are withint the unedited part.
        # what if it is also being edited?
        all_edits = []
        for i in self.features:
            n_edits = len(i.primitive.edit_sequence)
            all_edits.append(n_edits)
        min_edits = min(all_edits)
        if min_edits > 0:
            # Both have edits
            update = False
        else:
            # if all_edits[0] > all_edits[4]:
            #     edited_part = self.features[0]
            #     unedited_part = self.features[4]
            # else:
            #     edited_part = self.features[4]
            #     unedited_part = self.features[0]
            # expression = edited_part.dynamic_expression()
            # X = MAIN_VAR
            # points = expression.subs({X: TEST_VALUE})
            # # now get the local coord of this.
            # points = np.asarray(points).astype(np.float32)
            # obb = unedited_part.primitive
            # center = np.array(obb.center()).astype(np.float32)
            # size = np.array(obb.size).astype(np.float32)
            # axis = np.array(obb.axis).astype(np.float32)
            # local_coord = ((points - center) @ axis) / (size/2)
            # if np.any(np.abs(local_coord) > 1):
            #     update = False
            # else:
            #     update = True
            update = True # if it depends on numerical then dont
        return update
    
class HeightContact(FeatureRelation):
    def __init__(self, feature_group, relation_index):
        super().__init__(feature_group, relation_index)
        broken = self.broken()
        if broken:
            delta = self.get_discrepancy()
            initial_errors = []
            for i in range(delta.shape[0]):
                delta_vec = delta[i, :]#.norm()
                initial_errors.append(delta_vec)
            # print(initial_errors)
            self.initial_errors = initial_errors
        
    def broken(self):
        # we first need the motion vectors for each part.
        # Should we handle approximate GT -> Based on initial discrepancy?
        if len(self.edit_sequence) > 0:
            broken = False
        else:
            broken = False
            all_edits = []
            for i in self.features:
                n_edits = len(i.primitive.edit_sequence)
                all_edits.append(n_edits)
            min_edit = min(all_edits)
            if min_edit > 0:
                mode = 2
            else:
                mode = 1

            delta = self.get_discrepancy()
            i = 0
            delta_i = delta[i]
            if self.initial_errors is not None:
                delta_i = delta_i - self.initial_errors[i]
            equals_zero = evaluate_equals_zero(delta_i, mode=mode, order=3)
            if not equals_zero:
                broken = True
            return broken
        
    def get_discrepancy(self):
        part_1_motion = self.features[0].dynamic_expression()
        part_2_motion = self.features[1].dynamic_expression()
        delta = part_1_motion - part_2_motion
        delta = delta[:, 1:2]
        return delta

    def resolve(self):
        raise NotImplementedError()
    
    
    def prompt_signature(self):
        sig_template = f"Planar contact between ($point_str_1, $point_str_2)"
        point_str_1 = self.features[0].prompt_signature()
        point_str_2 = self.features[1].prompt_signature()
        sig_replace_dict = {"point_str_1": point_str_1, "point_str_2": point_str_2}
        sig = Template(sig_template).substitute(sig_replace_dict)
        return sig

    def updatable(self):
        # i.e. I can make edit to the relation to retain the relation.
        # there is an edited part, and an unedited part.
        # its updatable only if the dymanic expression of features from edited_part are withint the unedited part.
        # what if it is also being edited?
        update = False

        return update
    
class MatchingRotationGroup(Relation):

    def __init__(self, relation_set, relation_index):
        self.relation_set = relation_set
        self.relation_index = relation_index
        for item in relation_set:
            item._relations.add(self)
    
    def broken(self):
        n_edits = []
        for relation in self.relation_set:
            n_edits.append(len(relation.edit_sequence))
        
        unique_edits = set(n_edits)
        if len(unique_edits) > 1:
            broken = True
        else:
            broken = False
        return broken
    
    def resolve(self):
        return True

    def resolve(self):

        n_edits = []
        for relation in self.relation_set:
            n_edits.append(len(relation.edit_sequence))
        
        max_ind = np.argmax(n_edits)
        source_relation = self.relation_set[max_ind]
        all_new_edits = []
        for ind, relation in enumerate(self.relation_set):
            if ind == max_ind:
                continue
                
            for edit in source_relation.edit_sequence:
                new_edit = edit.copy(relation)
                all_new_edits.append(new_edit)
        return all_new_edits
    
    def updatable(self):
        return False
    
