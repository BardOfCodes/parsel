import copy

import numpy as np
import open3d as o3d
import torch as th

from .constants import COLORS
from ..data.geometry import obb_to_mesh
from ..shape_system.constants import PART_ACTIVE, PART_INACTIVE
from ..shape_system.shape import Part

INDEX_BUMP = 100


def create_geom_map(
    parts_dict,
    relations_dict,
    intersection_dict,
    add_color=True,
):
    """Build the geometry and 3D-label maps consumed by the visualizers."""
    name_to_geom = {}
    label_dict = {}
    for ind, part in parts_dict.items():
        name = f"{ind}_mesh"
        if add_color:
            for x in part['mesh'].meshes:
                x.mesh.paint_uniform_color(COLORS[ind])
        name_to_geom[name] = part['mesh']

        name = f"{ind}_obb"
        obb_mesh = obb_to_mesh(part['obb'])
        obb_mesh.compute_vertex_normals()
        normals = np.array(obb_mesh.triangle_normals)
        normals = -normals
        obb_mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
        obb_mesh.paint_uniform_color(COLORS[ind])
        name_to_geom[name] = obb_mesh

        label_dict[f"{ind}_3d_label"] = (part['label'], part['obb'].center, None)

    for ind, relation in relations_dict.items():
        if relation['type'] == "R":
            axis, origin = relation['params']
            relation_geom = create_arrow(origin, origin + axis)
        elif relation['type'] == "T":
            delta = relation['params'][0]
            part_ind = relation['indices'][0]
            part = parts_dict[part_ind]
            center = part['obb'].center
            relation_geom = create_arrow(center, center + delta)
        elif relation['type'] == "REF":
            plane_axis, plane_origin = relation['params']
            relation_geom = create_plane(plane_axis, plane_origin)
        else:
            raise ValueError(
                f"Unsupported visualization relation type: "
                f"{relation['type']!r}"
            )
        name = f"{ind}_R"
        name_to_geom[name] = relation_geom
        label_dict[name + "_3d_label"] = (relation['name'], relation_geom.get_center(), None)

    for ind, intersection in intersection_dict.items():
        name = f"{ind}_I"
        point_cloud = o3d.geometry.PointCloud()
        points = intersection['points']
        point_cloud.points = o3d.utility.Vector3dVector(points)
        color = np.array([1, 0, 0])
        color_set = np.tile(color, (points.shape[0], 1))
        point_cloud.colors = o3d.utility.Vector3dVector(color_set)
        name_to_geom[name] = point_cloud
        label_dict[name + "_3d_label"] = (intersection['name'], point_cloud.get_center(), None)
    return name_to_geom, label_dict


def create_switch_board(name_to_geom, label_dict):
    """Return an initially hidden switch for every geometry and label."""
    return dict.fromkeys((*name_to_geom, *label_dict), False)


def update_switch_board(name_to_geom, label_dict, shape):
    """Show active part meshes and labels; hide all auxiliary geometry."""
    parts_by_index = {part.part_index: part for part in shape.partset}
    switch_board = dict.fromkeys((*name_to_geom, *label_dict), False)

    for name in name_to_geom:
        prefix, _, item_type = name.partition("_")
        if item_type != "mesh":
            continue
        part = parts_by_index.get(int(prefix))
        if part is None:
            raise ValueError(f"Geometry {name!r} has no matching shape part")
        switch_board[name] = part.state[0] == PART_ACTIVE

    for name in label_dict:
        prefix, _, suffix = name.partition("_")
        if suffix != "3d_label":
            continue
        part = parts_by_index.get(int(prefix))
        if part is None:
            raise ValueError(f"Label {name!r} has no matching shape part")
        switch_board[name] = part.state[0] == PART_ACTIVE

    return switch_board


def _rotation_between(source, target):
    """Return a rotation matrix that aligns one normalized vector to another."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    dot = np.clip(np.dot(source, target), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        candidate = np.array([1.0, 0.0, 0.0])
        if np.isclose(abs(np.dot(source, candidate)), 1.0):
            candidate = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, candidate)
        axis /= np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis * np.pi
        )

    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def create_arrow(start, end):
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Arrow start and end must be distinct finite points")

    cylinder_length = 0.75 * length
    cone_length = 0.25 * length
    cylinder_radius = 0.02
    cone_radius = 0.04
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius, cone_radius, cylinder_length, cone_length)
    arrow.compute_vertex_normals()
    direction = direction / length
    rotation = _rotation_between(np.array([0.0, 0.0, 1.0]), direction)
    arrow.rotate(rotation, center=(0, 0, 0))
    arrow.translate(start)
    return arrow


def create_plane(normal, origin, size=1.0):
    normal = np.asarray(normal, dtype=float)
    normal_length = np.linalg.norm(normal)
    if not np.isfinite(normal_length) or normal_length <= 0:
        raise ValueError("Plane normal must be a non-zero finite vector")
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Plane size must be a positive finite number")
    normal /= normal_length

    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
    centering = np.array([size/2, 0.005, size/2])
    mesh.translate(-centering)
    rotation = _rotation_between(np.array([0.0, 1.0, 0.0]), normal)
    mesh.rotate(rotation, center=(0, 0, 0))
    mesh.translate(origin)
    mesh.compute_vertex_normals()
    return mesh


def update_model(
    shape,
    parts_dict,
    symbol_to_parts,
    symbol_value_dict,
    cube_coord_dict,
    name_to_geom,
    device=None,
):
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    multiplicity = 0
    for symbol in symbol_value_dict:
        relevant_parts = symbol_to_parts[symbol]
        for part in relevant_parts:
            if len(part.partset) > 0:
                # see which kind of execution is required.
                core_relation = part.core_relation
                output_bboxes = core_relation.execute_relation(symbol_value_dict)
                n_boxes = len(output_bboxes)
                if n_boxes > 1:
                    multiplicity += 1
                    parent_part = part
                    parent_part.sub_parts = {}
                    child = part.core_relation.primitives[0].part

                    original_index = child.part_index
                    original_part_dict = parts_dict[original_index]

                    cur_coords = cube_coord_dict[original_index]

                    new_index = len(shape.partset) + INDEX_BUMP * (multiplicity + 1)
                    new_children = []
                    output_bboxes = np.stack([np.asarray(x) for x in output_bboxes], axis=0).astype(np.float32)
                    output_bboxes = th.from_numpy(output_bboxes).to(device)
                    for index in range(n_boxes):
                        real_th = output_bboxes[index]

                        vertices = cur_coords[..., None] * real_th[None, ...]
                        vertices = vertices.sum(dim=1)

                        label = f"{child.original_label}_{index}"
                        cur_index = new_index + index
                        new_child = Part(
                            label,
                            child.primitive,
                            part_index=cur_index,
                            has_children=False,
                            original_label=child.original_label,
                        )
                        new_children.append(new_child)
                        # replicate part dict
                        original_tm = original_part_dict[f"mesh"]
                        new_tm = o3d.visualization.rendering.TriangleMeshModel()
                        new_tm.materials = [x for x in original_tm.materials]
                        new_meshes = []
                        vertice_count = 0
                        for mesh_info in original_tm.meshes:
                            cur_mesh = copy.deepcopy(mesh_info.mesh)
                            material_idx = mesh_info.material_idx
                            new_mesh_info = o3d.visualization.rendering.TriangleMeshModel.MeshInfo(cur_mesh, "", material_idx)
                            cur_n_vertices = len(mesh_info.mesh.vertices)
                            cur_vertices = vertices[vertice_count: vertice_count + cur_n_vertices]
                            cur_vertices = cur_vertices.cpu().numpy().astype(np.float64)
                            creation = o3d.utility.Vector3dVector(cur_vertices)
                            new_mesh_info.mesh.vertices = creation
                            new_meshes.append(new_mesh_info)
                            vertice_count += cur_n_vertices
                        new_tm.meshes = new_meshes
                        new_part_dict = dict(mesh=new_tm, obb=original_part_dict['obb'], label=child.original_label,
                                             objs=original_part_dict['objs'], )
                        parts_dict[cur_index] = new_part_dict
                        name_to_geom[f"{cur_index}_mesh"] = new_tm
                        # Other updates
                        cube_coord_dict[cur_index] = cube_coord_dict[original_index]
                    new_children = set(new_children)
                    parent_part.sub_parts = new_children
                    parent_part.state[0] = PART_INACTIVE

    return shape, parts_dict


def _update_part_mesh(
    part_id,
    part_eq,
    cube_coord_dict,
    symbol_value_dict,
    name_to_geom,
    device,
):
    mesh_name = f"{part_id}_mesh"
    cur_cube_coords = cube_coord_dict[part_id]
    realization = part_eq.subs(symbol_value_dict)
    real_np = np.asarray(realization).astype(np.float32)
    real_th = th.from_numpy(real_np).to(device)
    vertices = cur_cube_coords[..., None] * real_th[None, ...]
    vertices = vertices.sum(dim=1).cpu().numpy().astype(np.float64)
    corresponding_mesh = name_to_geom[mesh_name]

    vertice_count = 0
    for mesh_info in corresponding_mesh.meshes:
        cur_n_vertices = len(mesh_info.mesh.vertices)
        cur_vertices = vertices[vertice_count: vertice_count + cur_n_vertices]
        creation = o3d.utility.Vector3dVector(cur_vertices)
        mesh_info.mesh.vertices = creation
        vertice_count += cur_n_vertices
