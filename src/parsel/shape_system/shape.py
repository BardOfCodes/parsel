import numpy as np
import sympy as sp

from .math import assign_directions
from .annotations import sympy_vec_to_str, sympy_annotate_via_distance_vec
from .constants import RELATION_UNCHECKED, RELATION_RETAINED, PART_EDITED, PART_UNEDITED, PART_ACTIVE, PART_INACTIVE, RELATION_ACTIVE, RELATION_INACTIVE, UPDATE_RELATION


##################### Primitives #####################
# Always use format - (Number of points, Dimensions)
class Primitive:
    ...

class Hexahedron(Primitive):

    def __init__(self, point_set, obb=None):
        self.point_set = sp.Matrix(point_set) # shape 3 x 8
        self._features = set()
        self._relations = set()
        self.edit_sequence = []
        self.param_names = ["point_set"]
        self.obb = obb

        self.assign_directions()

    def create_interesting_points(self):

        # create interesting_point_set, point_names
        center_point = np.array(self.center()).astype(np.float32)
        interesting_point_set = [center_point]
        point_names = [f"{self.part.full_label}'s center"]
        for key, value in self.name_to_indices.items():
            if isinstance(key, str):
                point = self.face_center(key)
                np_point = np.array(point).astype(np.float32)
                point_names.append(f"center of the {key} face of {self.part.full_label}")
                interesting_point_set.append(np_point)
            elif len(key) == 2:
                point = self.edge_center(*key)
                np_point = np.array(point).astype(np.float32)
                point_names.append(f"center of the {key} edge of {self.part.full_label}")
                interesting_point_set.append(np_point)
            elif len(key) == 3:
                point = self.corner(*key)
                np_point = np.array(point).astype(np.float32)
                point_names.append(f"the {key} corner of {self.part.full_label}")
                interesting_point_set.append(np_point)
        self.interesting_point_set = interesting_point_set
        self.point_names = point_names
    
    def assign_directions(self):
        # Assign Faces
        self.name_to_indices = assign_directions(self.point_set)
        self.indices_to_name = {}
        for key, value in self.name_to_indices.items():
            if isinstance(key, str):
                self.indices_to_name[tuple(sorted(list(value)))] = key
            else:
                self.indices_to_name[tuple(sorted(list(value)))] = tuple(sorted(list(key)))
        # temp
        self.orient_keys = [
            ('right', 'up', 'front'),
            ('left', 'up', 'front'),# [-1, 1, 1],
            ('left', 'down', 'front'), # [-1, -1, 1],
            ('right', 'down', 'front'), # [1, -1, 1],
            ('right', 'up', 'back'), # [1, 1, -1],
            ('left', 'up', 'back'), # [-1, 1, -1],
            ('left', 'down', 'back'), # [-1, -1, -1],
            ('right', 'down', 'back'), # [1, -1, -1],
        ]
        self.oriented_pointset = sp.Matrix.vstack(*[self.point_set[self.name_to_indices[tuple(sorted(list(i)))], :] for i in self.orient_keys])



    @property
    def size(self):
        size_x = self.face_center("right") - self.face_center("left")
        size_y = self.face_center("up") - self.face_center("down")
        size_z = self.face_center("front") - self.face_center("back")
        size_x = size_x.norm()
        size_y = size_y.norm()
        size_z = size_z.norm()



        return sp.Matrix(1, 3, [size_x, size_y, size_z])
    
    @property
    def axis(self):
        # The axis is the direction of the longest edge.
        size_x = self.face_center("right") - self.face_center("left")
        size_y = self.face_center("up") - self.face_center("down")
        size_z = self.face_center("front") - self.face_center("back")
        size_x = size_x / size_x.norm()
        size_y = size_y / size_y.norm()
        size_z = size_z / size_z.norm()


        return sp.Matrix.vstack(size_x, size_y, size_z)

    @property
    def params(self):
        return {x:getattr(self, x) for x in self.param_names}
    
    @property
    def relations(self):
        return self._relations

    @property
    def features(self):
        return self._features
    
    def static_expression(self, oriented=False):
        if oriented:
            return self.oriented_pointset
        else:
            return self.point_set
    
    def dynamic_expression(self, oriented=False):
        if oriented:
            points = self.oriented_pointset
        else:
            points = self.point_set
        for motion in self.edit_sequence:
            points = motion.apply(points)
        return points

    def center(self):
        # mean of the points
        mean = sp.ones(1, self.point_set.shape[0]) * self.point_set / self.point_set.shape[0]
        return mean
    
    def face_center(self, direction):
        # Relabel directions if required.
        face_indices = self.name_to_indices[direction]
        points = self.point_set[face_indices, :]
        center = sp.ones(1, points.shape[0]) * points / points.shape[0]
        return center
    
    def edge_center(self, direction_1, direction_2):
        key = tuple(sorted([direction_1, direction_2]))
        edge_indices = self.name_to_indices[key]
        points = self.point_set[edge_indices, :]
        center = sp.ones(1, points.shape[0]) * points / points.shape[0]
        return center
    
    def corner(self, direction_1, direction_2, direction_3):
        key = tuple(sorted([direction_1, direction_2, direction_3]))
        corner_index = self.name_to_indices[key]
        point = self.point_set[corner_index, :]
        return point

    def direction(self, direction):
        """Returns the direction vector from the center to the specified face of the cuboid.
        """
        face_center = self.face_center(direction)
        dir_vec = (face_center - self.center()).normalized()
        return dir_vec
    
    def get_point_of_contact(self, part):
        # Implement when required
        raise NotImplementedError
    

    def __repr__(self):
        return f"Hexahedron of {self.part.label}"
    
    def signature(self):
        ...
    
    def full_signature(self):
        ...

    def prompt_signature(self):
        ...

    def get_all_interesting_points(self, with_names=False):
        if with_names:
            return self.interesting_point_set, self.point_names
        else:
            return self.interesting_point_set
    

##################### Features #####################

class PrimitiveFeature:
    def __init__(self, primitive):
        self.primitive = primitive
        primitive._features.add(self)
        self._relations = set()
        self.param_names = []
    
    @property
    def params(self):
        return {x:getattr(self, x) for x in self.param_names}
    
    def static_expression(self):
        ...
    
    def dynamic_expression(self):
        ...

    def __repr__(self):
        param_str_list = [f"{x}={sympy_vec_to_str(y)}" for x, y in self.params.items()]
        param_str = ', '.join([x for x in param_str_list])
        label = self.primitive.part.label
        string = f"{self.__class__.__name__}(part={label}, {param_str})"
        return string
    
    def signature(self):
        return str(self)
    
    def full_signature(self):
        return str(self)
        
    @property
    def relations(self):
        return self._relations
    
    
class PointFeature(PrimitiveFeature):
    def __init__(self, primitive, harmonic_coords):
        super().__init__(primitive=primitive)
        self.harmonic_coords = sp.Matrix(harmonic_coords) # 1 x 8
        self.param_names = ["harmonic_coords"]
    
    def static_expression(self):
        end_points = self.primitive.static_expression() # 8, 3
        global_point = self.harmonic_coords * end_points
        return global_point
    
    def dynamic_expression(self):
        end_points = self.primitive.dynamic_expression()
        global_point = self.harmonic_coords * end_points
        return global_point
    
    def __repr__(self):
        # point = self.geometric_expression()
        # point = cube_to_local(np.asarray(self.harmonic_coords))
        # point_str = sympy_annotate_via_distance_vec(point)
        # Can use the naming scheme
        label = self.primitive.part.label
        # string = f"{self.__class__.__name__}(part={label}, close_to={point_str})"
        cur_string = f"Point({label}, {' '.join([f'{float(x):.2f}' for x in self.harmonic_coords])}"
        return cur_string
    
    def prompt_signature(self):
        # point = self.geometric_expression()
        point = self.relative_position
        point_str = sympy_annotate_via_distance_vec(point)
        label = self.primitive.part.label
        string = f"point near {label}'s {point_str}"
        return string

    def signature(self):
        # point = self.geometric_expression()
        point = self.relative_position
        point_str = sympy_vec_to_str(point)
        # Can use the naming scheme
        label = self.primitive.part.label
        string = f"{self.__class__.__name__}(part={label}, relative_position={point_str})"
        return string
    
    def full_signature(self):
        raise NotImplementedError
    
class Part:
    def __init__(self, label: str, primitive: Primitive, 
                 sub_parts: set = None, part_index=None,
                 original_label=None,
                 has_children=0):
        self.label = label
        self.primitive = primitive
        self.indices_to_name = primitive.indices_to_name
        self.name_to_indices = primitive.name_to_indices
        self.primitive.part = self
        if sub_parts is None:
            sub_parts = set()
        self.new_parts = set()
        self.parent = None
        self.sub_parts = sub_parts
        self.state = [PART_ACTIVE, PART_UNEDITED, has_children]
        self.part_index = part_index
        self.original_label = original_label
    # first order relations only
    # all relations
    def primitive_relations(self, get_all=False):
        relations = self.primitive.relations
        if get_all:
            for part in self.sub_parts:
                relations = relations.union(part.primitive_relations(get_all=get_all))
        return relations
    
    def active_primitive_relations(self, get_all=False):
        relations = set([x for x in self.primitive.relations if x.state[0] == RELATION_ACTIVE])
        if get_all:
            for part in self.sub_parts:
                relations = relations.union(part.active_primitive_relations(get_all=get_all))
        return relations
    
    def feature_relations(self, get_all=False):
        relations = set()
        for feature in self.primitive.features:
            relations = relations.union(feature.relations)
        if get_all:
            for part in self.sub_parts:
                relations = relations.union(part.feature_relations(get_all=get_all))
        return relations

    def active_feature_relations(self, get_all=False):
        relations = set()
        for feature in self.primitive.features:
            relations = relations.union([x for x in feature.relations if x.state[0] == RELATION_ACTIVE])
        if get_all:
            for part in self.sub_parts:
                relations = relations.union(part.active_feature_relations(get_all=get_all))
        return relations

    def all_relations(self, only_active=False):
        if only_active:
            relation_set = self.active_primitive_relations(get_all=True).union(self.active_feature_relations(get_all=True))
        else:
            relation_set = self.primitive_relations(get_all=True).union(self.feature_relations(get_all=True))
        return relation_set
    
    @property
    def partset(self):
        partset = set()
        for part in self.sub_parts:
            partset.add(part)
        for part in self.sub_parts: 
            partset = partset.union(part.partset)
        return partset
    
    @property
    def full_label(self):
        if self.parent is None:
            return self.label
        else:
            return f"{self.parent.full_label}/{self.label}"
    
    @property
    def new_partset(self):
        partset = set()
        for part in self.new_parts:
            partset.add(part)
        for part in self.sub_parts: 
            partset = partset.union(part.new_partset)
        return partset
    
    def features(self, get_all=False):
        features = set()
        for feature in self.primitive.features:
            features.add(feature)
        if get_all:
            for part in self.sub_parts:
                features = features.union(part.features(get_all=True))
        return features
    
    def primitives(self, get_all=False):
        primitives = set()
        primitives.add(self.primitive)
        if get_all:
            for part in self.sub_parts:
                primitives = primitives.union(part.primitives(get_all=True))
        return primitives

    @property
    def leaf_primitives(self):
        primitives = set()
        if len(self.sub_parts) == 0:
            primitives.add(self.primitive)
        else:
            for part in self.sub_parts:
                primitives = primitives.union(part.leaf_primitives)
        return primitives
    
    def get(self, label):
        """Find a part by a unique label or an explicit hierarchy path."""
        if "(" in label:
            label = label.split("(")[0].strip()
        components = [item for item in label.split("/") if item]
        if components and components[0] == self.label:
            components = components[1:]

        if not components:
            if label == self.label:
                return self
            raise ValueError(f"Part with label {label!r} not found.")

        if len(components) > 1:
            current = self
            for component in components:
                matches = [
                    part
                    for part in current.sub_parts
                    if part.label == component
                ]
                if len(matches) != 1:
                    raise ValueError(f"Part path {label!r} not found.")
                current = matches[0]
            return current

        target = components[0]
        matches = [
            part
            for part in ({self} | self.partset)
            if part.label == target
        ]
        if not matches:
            raise ValueError(f"Part with label {target!r} not found.")
        if len(matches) > 1:
            paths = ", ".join(
                sorted(part.full_label for part in matches)
            )
            raise ValueError(
                f"Part label {target!r} is ambiguous; use one of: {paths}"
            )
        return matches[0]
    

    def get_relation(self, label):
        part = self.get(label)
        if hasattr(part, "core_relation"):
            relation = part.core_relation
        else:
            relation = None
        return relation


    def clean_up_motion(self):
        self.primitive.edit_sequence = []
        for part in self.sub_parts:
            part.clean_up_motion()
        if hasattr(self, "core_relation"):
            self.core_relation.edit_sequence = []
    
    def center(self):
        return self.primitive.center()

    def face_center(self, face_type):
        return self.primitive.face_center(face_type)
    
    def direction(self, direction):
        return self.primitive.direction(direction)
    
    def corner(self, direction_1, direction_2, direction_3):
        return self.primitive.corner(direction_1, direction_2, direction_3)
    
    def edge_center(self, direction_1, direction_2):
        return self.primitive.edge_center(direction_1, direction_2)
    
    def get_point_of_contact(self, part):
        return self.primitive.get_point_of_contact(part)
    
    def __repr__(self):
        cur_str = f"Part(label={self.label})"
        return cur_str
    
    def prompt_signature(self):
        return f"{self.label}"
    
    def signature(self):
        return str(self)
    
    def full_signature(self):
        raise NotImplementedError

    def deactivate_parent(self, part):
        to_be_deactivated = part.part_index
        to_be_activated = [child.part_index for child in part.sub_parts]
        part.state[0] = PART_INACTIVE
        from .relations import PrimitiveRelation
        for child in part.sub_parts:
            child.state[0] = PART_ACTIVE
        for relation in part.all_relations(only_active=False):
            if isinstance(relation, PrimitiveRelation):
                relevant_indices = [x.part.part_index for x in relation.primitives]
            else:
                relevant_indices = [x.primitive.part.part_index for x in relation.features]
            if to_be_deactivated in relevant_indices:
                relation.state[0] = RELATION_INACTIVE
            elif any([x in relevant_indices for x in to_be_activated]):
                relation.state[0] = RELATION_ACTIVE
            
    def face(self, name):
        return FaceFeature(self.primitive, name)
    
    def edge(self, direction_1, direction_2):
        name = tuple(sorted([direction_1, direction_2]))
        return EdgeFeature(self.primitive, name)

    def get_all_interesting_points(self, with_names=True):
        # Collect from all edited parts
        all_points = []
        all_point_names = []
        all_edits = self.get_all_edits()
        # Collect from all validated relations
        from .relations import PrimitiveRelation
        for edit in all_edits:
            interesting_points, point_names = edit.operand.primitive.get_all_interesting_points(with_names=True)
            all_points.extend(interesting_points)
            all_point_names.extend(point_names)
        
        relations = self.all_relations(only_active=True)
        retained_relations = [x for x in relations if x.state[2] == RELATION_RETAINED]
        # now from these only the ones that have to be resolved - not the ones that must be updated
        retained_relations = [x for x in retained_relations if x.state[3] != UPDATE_RELATION]
        for relation in retained_relations:
            interesting_points, point_names = relation.get_all_interesting_points(with_names=True)
            all_points.extend(interesting_points)
            all_point_names.extend(point_names)
        all_points = np.concatenate(all_points, axis=0)
        if with_names:
            return all_points, all_point_names
        else:
            return all_points

    def get_all_edits(self):
        all_edits = []
        for part in self.partset:
            all_edits.extend(part.primitive.edit_sequence)
        return all_edits


class FaceFeature:
    def __init__(self, primitive, name):
        self.primitive = primitive
        self.name = name
class EdgeFeature:

    def __init__(self, primitive, name):
        self.primitive = primitive
        self.name = name
