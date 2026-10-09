import os
import open3d as o3d
import numpy as np
import json
import copy

from .constants import SCALE_FACTOR, PC_SIZE, MIN_SCALE_DELTA, MAX_SCALE_DELTA
from .geometry import get_oriented_bounding_box_with_fixing

def add_mesh_only(mesh_dir, parts, obb_o3d=False):
    for part_dict in parts:
        mesh_files = part_dict['objs']
        for ind, mesh_file in enumerate(mesh_files):
            if '.' not in mesh_file:
                mesh_files[ind] = f"{mesh_file}.obj"
        # mesh_files = [os.path.join(mesh_dir, f"{f}.obj") for f in mesh_files]
        mesh_files = [os.path.join(mesh_dir, f"{f}") for f in mesh_files]
        # meshes = [o3d.io.read_triangle_mesh(f) for f in mesh_files]
        meshes = [o3d.io.read_triangle_model(f) for f in mesh_files]
        # join the meshes
        if len(meshes) > 1:
            mesh = meshes[0]
            for m in meshes[1:]:
                mesh.meshes = mesh.meshes + m.meshes
                mesh.materials = mesh.materials + m.materials
        else:
            mesh = meshes[0]
        # get oriented bounding box
        part_dict['mesh'] = mesh
    return parts

def enrich_data(model_id, model, clean_relations, 
                intersections, data_dir, mode="compat"):
    data_dir = os.path.join(data_dir, model_id, "objs")
    add_color = True
    # change the 
    for ind, part in enumerate(model):
        center, extent, R = part['obb']
        part['obb'] = o3d.geometry.OrientedBoundingBox(center, R, extent)
        # load the mesh

        # mesh = o3d.geometry.TriangleMesh()
        mesh = o3d.visualization.rendering.TriangleMeshModel()

        for obj_file in part['objs']:
            # obj_file = os.path.join(data_dir, f"{obj_file}.obj")
            if "." not in obj_file:
                obj_file = f"{obj_file}.obj"
            obj_file = os.path.join(data_dir, f"{obj_file}")
            # cur_mesh = o3d.io.read_triangle_mesh(obj_file)
            cur_mesh = o3d.io.read_triangle_model(obj_file)
            # cur_mesh = mesh_cleanup(cur_mesh)
            # cur_mesh.compute_vertex_normals()
            # mesh += cur_mesh
            # if mode == "compat":
            #     original_glb_mesh = trimesh.load_mesh(obj_file)
            #     for ind, curcur_mesh in enumerate(cur_mesh.meshes):
            #         geom =  original_glb_mesh.geometry[f'geometry_{ind}']
            #         curcur_mesh.mesh.vertices = o3d.utility.Vector3dVector(geom.vertices)
            #         curcur_mesh.mesh.triangles = o3d.utility.Vector3iVector(geom.faces)
            #         curcur_mesh.mesh.vertex_normals = o3d.utility.Vector3dVector(geom.vertex_normals)

            mesh.meshes = mesh.meshes + cur_mesh.meshes
            mesh.materials = mesh.materials + cur_mesh.materials
            # Update the vertices here with trimesh
            # FIX for o3d and trimesh mismatch

        # paint mesh
        # mesh.compute_vertex_normals()
        
        part['mesh'] = mesh
        part['name'] = f"{ind}_{part['label']}"
    
    # let the representation be a dict
    part_dict = {ind: part for ind, part in enumerate(model)}
    # For relations get things: 
    relation_dict = {}
    for rel_ind, relation in enumerate(clean_relations):
        cur_dict = {}
        cur_dict["indices"] = relation[0]
        part_name = [part_dict[ind]['label'] for ind in relation[0]]
        cur_dict['part_names'] = part_name
        cur_dict['name'] = f"{relation[-1]}_{relation[0]}"
        cur_dict['params'] = relation[1]
        cur_dict['type'] = relation[-1]
        relation_dict[rel_ind] = cur_dict

    intersection_dict = {}
    for ind, intersection in enumerate(intersections):
        cur_dict = {}
        ind_1, ind_2, point_set, order = intersection
        part_name_1 = part_dict[ind_1]['label']
        part_name_2 = part_dict[ind_2]['label']
        cur_dict['name'] = f"Int_{ind_1}_{ind_2}_{order}"
        cur_dict['indices'] = (ind_1, ind_2)
        cur_dict['part_names'] = (part_name_1, part_name_2)
        cur_dict['points'] = point_set
        cur_dict['order'] = order
        intersection_dict[ind] = cur_dict
    
    return part_dict, relation_dict, intersection_dict, add_color
def mesh_cleanup(mesh_model):
    for mesh_info in mesh_model.meshes:
        mesh = mesh_info.mesh
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        mesh_info.mesh = mesh
    return mesh_model

def mesh_to_t_mesh(mesh):
    # If you have more get more.
    device = o3d.core.Device("CPU:0")
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32
    vertices = np.array(mesh.vertices, dtype=np.float32)
    triangles = np.array(mesh.triangles, dtype=np.int32)
    t_mesh = o3d.t.geometry.TriangleMesh(device)
    t_mesh.vertex.positions = o3d.core.Tensor(vertices, device=device, dtype=dtype_f)
    t_mesh.triangle.indices = o3d.core.Tensor(triangles, device=device, dtype=dtype_i)
    return t_mesh

def load_parts(model_id, data_dir, max_depth=2):
    data_file = os.path.join(data_dir, model_id, "result_after_merging.json")

    # STEP 1: load content
    parts = []
    with open(data_file, "r") as f:
        data = json.load(f)
    depth = 0
    stack = []
    for child in data:
        stack.append((child, depth, None))
    while len(stack) > 0:
        node, depth, parent_carry_name = stack.pop()
        if parent_carry_name is not None:
            label = parent_carry_name
        else:
            label = node['name']
        if depth > max_depth:
            part_dict = {
                'objs': node["objs"],
                'label': label, # node['name'],
                'original_label': label, # node['name'],
                'hier': 0,
                'children': [],
            }
            parts.append(part_dict)
        else:
            if "children" in node.keys() and len(node["children"]) > 0:
                if len(node["children"]) == 1:
                    delta = 0
                    if parent_carry_name is not None:
                        carry_name = parent_carry_name
                    else:
                        carry_name = node['name']
                else:
                    delta = 1
                    carry_name = None
                for child in node["children"]:
                    stack.append((child, depth + delta, carry_name))
            else:

                part_dict = {
                    'objs': node["objs"],
                    'label': label, # node['name'],
                    'original_label': label, # node['name'],
                    'hier': 0,
                    'children': [],
                }
                parts.append(part_dict)
    print(f"Number of parts: {len(parts)}")
    category_label = data[0]['name']
    return parts, category_label


def add_obb_and_mesh(mesh_dir, parts, obb_o3d=False):
    for part_dict in parts:
        mesh_files = part_dict['objs']
        for ind, mesh_file in enumerate(mesh_files):
            if '.' not in mesh_file:
                mesh_files[ind] = f"{mesh_file}.obj"
        # mesh_files = [os.path.join(mesh_dir, f"{f}.obj") for f in mesh_files]
        mesh_files = [os.path.join(mesh_dir, f"{f}") for f in mesh_files]
        meshes = [o3d.io.read_triangle_mesh(f) for f in mesh_files]
        # meshes = [o3d.io.read_triangle_model(f) for f in mesh_files]
        # join the meshes
        if len(meshes) > 1:
            mesh = meshes[0]
            for m in meshes[1:]:
                mesh += m
        else:
            mesh = meshes[0]
        # get oriented bounding box
        mesh.compute_vertex_normals()
        # I think the scaling factor should be set such that it covers the mesh, but not extend too much.
        # instead of scaling up, get the bbox, and increase extents?
        temp_mesh = copy.deepcopy(mesh)# .scale(SCALE_FACTOR, mesh.get_center())

        # pcd = temp_mesh.sample_points_poisson_disk(number_of_points=PC_SIZE)
        pcd = temp_mesh.sample_points_uniformly(number_of_points=PC_SIZE)
        # add the vertices of the mesh to the pcd
        pcd.points.extend(mesh.vertices)
        
        if obb_o3d:
            obb = temp_mesh.get_oriented_bounding_box(robust=True)
        else:
            # COMPAS cause O3D SUCKS!
            obb = get_oriented_bounding_box_with_fixing(pcd)
        add_max = np.array([MAX_SCALE_DELTA,] * 3)
        add_min = np.array([MIN_SCALE_DELTA,] * 3)
        obb.extent = np.maximum(obb.extent * SCALE_FACTOR, obb.extent + add_min)
        obb.extent = np.minimum(obb.extent, obb.extent + add_max)
        part_dict['obb'] = obb
        part_dict['mesh'] = mesh
    return parts


def get_save_format(parts):
    resolved_parts = []
    for part in parts:
        obb = part['obb']
        resolved_part = dict()
        resolved_part['label'] = part['label']
        resolved_part['obb'] = (obb.get_center(), obb.extent, obb.R)
        resolved_part['hier'] = part['hier']
        resolved_part['original_label'] = part['original_label']
        resolved_part['objs'] = part['objs']
        resolved_part['children'] = part['children']
        resolved_parts.append(resolved_part)
    return resolved_parts
