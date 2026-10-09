# partnet does not have texture. 
# Step 1: load the file
# Step 2: file -> leaf nodes -> point cloud + bbox.
# step 3: sym candidates. 
# step 3: contact relations. 

import os
import json
import copy
import open3d as o3d
from hausdorff import hausdorff_distance
from collections import defaultdict
import open3d.visualization as vis
import numpy as np
import itertools
import networkx as nx
import torch as th
from scipy import optimize
import pytorch3d.transforms
from .geometry import (obb_to_mesh, get_rot_trans_candidates, invert_matrix_to_components, 
                    get_oriented_bounding_box_with_fixing, get_reflection_matrix,
                    line_close, axis_close, extract_normals_and_sizes)
from ._labels import relabel_parts
from .mesh import load_parts, add_obb_and_mesh
from .constants import (SCALE_FACTOR, PC_SIZE, MIN_VOLUME,ROT_HAUS_DIST_THRESHOLD,
                        HAUS_DIST_TRHESHOLD, ROT_AMOUNT_THRESHOLD, TRANS_AMOUNT_THRESHOLD,
                        REF_AXIS_ALIGNMENT_THRESHOLD,
                        ICP_THRESHOLD, ICP_FITNESS_THRESHOLD, MAX_ICP_ITER, INTER_TO_POINT_RATIO_FACTOR,
                        MIN_SIZE_RELATION_GROUP, DIV_THRESHOLD, REF_DIV_THRESHOLD)

def main_loader(model_id, data_dir, relabel=True):

    parts, category_label = load_parts(model_id, data_dir)
    all_labels = set([p['label'] for p in parts])
    parts_with_ids = [(i, p) for i, p in enumerate(parts)]
    mesh_dir = os.path.join(data_dir, str(model_id), "objs")
    parts = add_obb_and_mesh(mesh_dir, parts)
    label_to_syms = get_all_sym_relations(parts_with_ids)
    label_to_sel_syms = prune_sym_relations(label_to_syms, parts_with_ids, min_size_relation_group=3)
    intersections_indices = get_intersections(parts_with_ids, label_to_sel_syms)
    clean_relation_set = cleanup_relations(parts, label_to_sel_syms)
    # reset order based on relations
    # intersections_indices = update_intersection_by_relations(intersections_indices, clean_relation_set)
    add_new = True
    hier = 0
    while(add_new):
        prev_len = len(parts)
        parts, clean_relation_set, intersections_indices = update_hierarchy(parts, clean_relation_set, intersections_indices)
        new_len = len(parts)
        if prev_len == new_len:
            add_new = False
        else:
            hier += 1
    # group by hier
    intersections_indices = convert_intersection_to_points(parts, intersections_indices)
    # hier_parts = [x for x in parts if x['hier'] == 0]
    part_graph, index_redux = hier_tree(parts)
    # index redux
    for ind, relation in enumerate(clean_relation_set):
        indices, params, sym_group_type = relation
        new_indices = []
        for index in indices:
            new_indices.append(index_redux[index])
        relation = (new_indices, params, sym_group_type)
        clean_relation_set[ind] = relation
    for ind, intersection in enumerate(intersections_indices):
        index_a, index_b, sel_points, order = intersection
        # create point set
        intersections_indices[ind] = (index_redux[index_a], index_redux[index_b], sel_points, order)
    for part in parts:
        part['children'] = tuple([index_redux[x] - 1 for x in part['children']])

    if relabel:
        relabel_parts(part_graph, clean_relation_set)
    # convert format to something easier
    # convert format to something easier
    parts = []
    for node_id in part_graph.nodes():
        part = part_graph.nodes[node_id]['part']
        parts.append(part)
    parts = parts[1:]
    for ind, relation in enumerate(clean_relation_set):
        indices, params, sym_group_type = relation
        new_indices = []
        for index in indices:
            new_indices.append(index - 1)
        relation = (new_indices, params, sym_group_type)
        clean_relation_set[ind] = relation
    for ind, intersection in enumerate(intersections_indices):
        index_a, index_b, sel_points, order = intersection
        # create point set
        intersections_indices[ind] = (index_a - 1, index_b - 1, sel_points, order)

    return parts, clean_relation_set, intersections_indices, category_label
    

def hier_tree(parts):
    graph = nx.DiGraph()
    graph.add_node(0, part={})
    stack = [(part, 0) for part in parts if part['hier'] == 0]
    index_redux = dict()
    while(len(stack) > 0):
        cur_part, parent = stack[0]
        stack = stack[1:]
        cur_id = len(graph.nodes())
        graph.add_node(cur_id, part=cur_part)
        # add edge
        graph.add_edge(parent, cur_id)
        prev_index = parts.index(cur_part)
        index_redux[prev_index] = cur_id
        for child in cur_part['children']:
            stack.append((parts[child], cur_id))
    return graph, index_redux


def get_valid_normals(obb_1, obb_2):
    
    points = np.asarray(obb_2.get_box_points())
    obb_2_pcd = o3d.geometry.PointCloud()
    obb_2_pcd.points = o3d.utility.Vector3dVector(points)
    # Convert to bbox?

    center = obb_1.get_center()
    normals, sizes = extract_normals_and_sizes(obb_1)
    valid_normal_set = []
    valid_size_set = []
    valid_inds = []
    for ind, cur_nor in enumerate(normals):
        transform_mat = get_reflection_matrix(center, cur_nor)
        transfromed_points = copy.deepcopy(obb_2_pcd).transform(transform_mat)
        h_dist = hausdorff_distance(np.asarray(transfromed_points.points),
                                    np.asarray(obb_2_pcd.points))
        
        if h_dist < HAUS_DIST_TRHESHOLD:
            if h_dist < sizes[ind] * 0.5:
                valid_normal_set.append(cur_nor)
                valid_size_set.append(sizes[ind])
                valid_inds.append(ind)
    return valid_normal_set, valid_size_set, valid_inds

def convert_intersection_to_points(parts, intersection_indices):
    
    for ind, contact_set in enumerate(intersection_indices):
        id_1, id_2, sel_points, order = contact_set
        part_dict_1 = parts[id_1]
        part_dict_2 = parts[id_2]
        obb_1 = part_dict_1['obb']
        obb_2 = part_dict_2['obb']
        ratio_factor = INTER_TO_POINT_RATIO_FACTOR
        points = []
        normal_1, size_1, inds_1 = get_valid_normals(obb_1, obb_2)
        normal_2, size_2, inds_2 = get_valid_normals(obb_2, obb_1)
        found_order = min(len(normal_1), len(normal_2))
        if found_order != order:
            print("WHAT?")
        if order == 0:
            # point contact just get center of selpoints.
            point1 = np.mean(sel_points, axis=0)
            points.append(point1)

        elif order == 1:
            # get the line and make it ratio_factor% the length.
            # min is 1, one of them might have 2
            if len(normal_1) == 1:
                normal = normal_1[0]
            else:
                normal = normal_2[0]
            
            center = np.mean(sel_points, axis=0)
            centered_points = sel_points - center
            dot_prods = (normal[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_val, max_val = np.min(dot_prods), np.max(dot_prods)

            point_1 = center + normal * min_val * ratio_factor
            point_2 = center + normal * max_val * ratio_factor
            points.append(point_1)
            points.append(point_2)
            # points.append(point1)
        elif order == 2:
            # get a face (3 or 4 coords?) and make it ratio_factor% the area.
            # Both should have at least 2
            n_1 = len(normal_1)
            n_2 = len(normal_2)
            center = np.mean(sel_points, axis=0)
            if n_1 == 2:
                # create 4 points in the plane
                normal_1, normal_2 = normal_1[0], normal_1[1]
            else:
                # create 4 points in the plane
                normal_1, normal_2 = normal_2[0], normal_2[1]
            # size cannot be trusted - project sel points on the plane and get the size.
            centered_points = sel_points - center
            dot_prods = (normal_1[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_1, max_1 = np.min(dot_prods), np.max(dot_prods)
            dot_prods = (normal_2[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_2, max_2 = np.min(dot_prods), np.max(dot_prods)
            point_1 = center + normal_1 * max_1 * ratio_factor + normal_2 * max_2 * ratio_factor
            point_2 = center + normal_1 * max_1 * ratio_factor + normal_2 * min_2 * ratio_factor
            point_3 = center + normal_1 * min_1 * ratio_factor + normal_2 * max_2 * ratio_factor
            point_4 = center + normal_1 * min_1 * ratio_factor + normal_2 * min_2 * ratio_factor
            points.append(point_1)
            points.append(point_2)
            points.append(point_3)
            points.append(point_4)
        elif order == 3:
            # create tetrahedron and make it ratio_factor% the volume.
            center = np.mean(sel_points, axis=0)
            n_1 = len(normal_1)
            n_2 = len(normal_2)
            if n_1 == 3:
                # create 4 points in the plane
                normal_1, normal_2, normal_3 = normal_1[0], normal_1[1], normal_1[2]
            elif n_2 == 3:
                # create 4 points in the plane
                normal_1, normal_2, normal_3 = normal_2[0], normal_2[1], normal_2[2]
            else:
                normal_1 = np.array([1, 0, 0])
                normal_2 = np.array([0, 1, 0])
                normal_3 = np.array([0, 0, 1])

            centered_points = sel_points - center
            dot_prods = (normal_1[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_1, max_1 = np.min(dot_prods), np.max(dot_prods)
            dot_prods = (normal_2[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_2, max_2 = np.min(dot_prods), np.max(dot_prods)
            dot_prods = (normal_3[None, :] * centered_points).sum(axis=1, keepdims=True)
            min_3, max_3 = np.min(dot_prods), np.max(dot_prods)
            point_1 = center + normal_1 * max_1 * ratio_factor + normal_2 * max_2 * ratio_factor
            point_2 = center + normal_1 * max_1 * ratio_factor + normal_2 * min_2 * ratio_factor
            point_3 = center + normal_3 * min_3 * ratio_factor + normal_2 * max_2 * ratio_factor
            point_4 = center + normal_1 * min_1 * ratio_factor + normal_2 * min_2 * ratio_factor
            points.append(point_1)
            points.append(point_2)
            points.append(point_3)
            points.append(point_4)

        points = np.array(points)
        intersection_indices[ind] = (id_1, id_2, points, order)
        # intersection_indices[ind] = {"indices": (id_1, id_2), 
        #                              "points": points,
        #                              "order": order}
    return intersection_indices


def update_hierarchy(parts, clean_relations, intersections_indices):
    
    n_parts = len(parts)
    
    new_parts = get_minimal_sym_merge(parts, clean_relations)
    # for the new parts try the other functions
    if new_parts:
        parts_with_ids = {i + n_parts: p for i, p in enumerate(new_parts)}
        label_to_syms = get_all_sym_relations(parts_with_ids)
        label_to_sel_syms = prune_sym_relations(label_to_syms, parts_with_ids, min_size_relation_group=3)
        new_relations = cleanup_relations(parts_with_ids, label_to_sel_syms)
        # new_rels = []
        # for relations in new_relations:
        #     new_indices = tuple([x + n_parts for x in relations[0]])
        #     new_relation = (new_indices, relations[1], relations[2])
        #     new_rels.append(new_relation)
        # new_relations = new_rels
        # new_label_to_sel_syms = {}
        # for label, relation_list in label_to_sel_syms.items():
        #     new_relation_list = []
        #     for relation in relation_list:
        #         new_indices = [x + n_parts for x in relation[0]]
        #         new_relation = (new_indices, relation[1])
        #         new_relation_list.append(new_relation)
        #     new_label_to_sel_syms[label] = new_relation_list
        # label_to_sel_syms = new_label_to_sel_syms


        # prune based on "real Divergence"
        # Update the indices in new relations and intersections 

        # Here we need to adjust the indices
        all_intersections = []
        for ind, new_part in parts_with_ids.items():
            cur_parts_with_ids = [(ind, new_part),]
            children_indices= new_part['children']
            prev_parts_with_inds = [(i, p) for i, p in enumerate(parts) if i not in children_indices]
            new_intersections = get_intersections(cur_parts_with_ids, label_to_sel_syms, prev_parts_with_inds)
            new_intersections = update_intersection_by_relations(new_intersections, new_relations)
            all_intersections.extend(new_intersections)
        # now between the new siblings
        cur_parts_with_ids = [(x, y) for x, y in parts_with_ids.items()]
        new_intersections = get_intersections(cur_parts_with_ids, label_to_sel_syms)
        new_intersections = update_intersection_by_relations(new_intersections, new_relations)
        all_intersections.extend(new_intersections)
        # add delta to every thing:
        # create a dict
        # propagate reflection of T or R down.
        # No Ref or ref possible as we don't group based on ref.
        under_lying_relations = []
        for relation in new_relations:
            indices, params, sym_group_type = relation
            if sym_group_type == "REF":
                group_1, group_2 = indices
                relation_1 = parts_with_ids[group_1]['relation_ind']
                relation_2 = parts_with_ids[group_2]['relation_ind']
                indices_1, params_1, sym_group_type_1 = clean_relations[relation_1]
                indices_2, params_2, sym_group_type_2 = clean_relations[relation_2]
                if len(indices_1) != len(indices_2):
                    continue
                # sort each entry in invalid_subsets:
                    # The top bbox can be wrong -
                valid_subsets = [] 
                list_A = [x for x in indices_1]
                list_B = [x for x in indices_2]
                for index_1 in list_A:
                    part = parts[index_1]
                    obb_1 = part['obb']
                    center_1 = obb_1.get_center()
                    plane_normal, origin = params
                    reflected_center_1 = center_1 - 2 * np.dot(plane_normal, center_1 - origin) * plane_normal
                    remaining_other_centers = np.array([parts[index_2]['obb'].get_center() for index_2 in list_B])
                    min_dist = [np.linalg.norm(reflected_center_1 - x) for x in remaining_other_centers]
                    min_ind = np.argmin(min_dist)
                    valid_subsets.append((index_1, list_B[min_ind]))
                    list_B.pop(min_ind)
                for subset in valid_subsets:
                    new_relation = (subset, tuple([x.copy() for x in params]), sym_group_type)
                    under_lying_relations.append(new_relation)
                    # for subset in valid_subsets:
                    #     cur_parts_with_ids = [(i, parts[i]) for i in subset]
                    #     cur_label_to_syms = get_all_sym_relations(cur_parts_with_ids)
                    #     cur_label_to_sel_syms = prune_sym_relations(cur_label_to_syms, cur_parts_with_ids)
                    #     cur_new_relations = cleanup_relations(parts, cur_label_to_sel_syms)
                    #     new_relations.extend(cur_new_relations)

                    
            elif sym_group_type == "T":
                # Its R/T of a R/T
                ...
                # group_1, group_2 = indices
                # relation_1 = new_parts[group_1]['relation_ind']
                # relation_2 = new_parts[group_2]['relation_ind']
                # indices_1, params_1, sym_group_type_1 = clean_relations[relation_1]
                # indices_2, params_2, sym_group_type_2 = clean_relations[relation_2]
                # if len(indices_1) != len(indices_2):
                #     continue
                # # sort each entry in invalid_subsets:
                #     # The top bbox can be wrong -
                # valid_subsets = [] 
                # list_A = [x for x in indices_1]
                # list_B = [x for x in indices_2]
                # for index_1 in list_A:
                #     part = parts[index_1]
                #     obb_1 = part['obb']
                #     center_1 = obb_1.get_center()
                #     plane_normal, origin = params
                #     reflected_center_1 = center_1 - 2 * np.dot(plane_normal, center_1 - origin) * plane_normal
                #     remaining_other_centers = np.array([parts[index_2]['obb'].get_center() for index_2 in list_B])
                #     min_dist = [np.linalg.norm(reflected_center_1 - x) for x in remaining_other_centers]
                #     min_ind = np.argmin(min_dist)
                #     valid_subsets.append((index_1, list_B[min_ind]))
                #     list_B.pop(min_ind)
                # for subset in valid_subsets:
                #     new_relation = (subset, tuple([x.copy() for x in params]), sym_group_type)
                #     under_lying_relations.append(new_relation)
            else:
                print("What")
        new_relations.extend(under_lying_relations)
        parts.extend(new_parts)
        clean_relations.extend(new_relations)
        intersections_indices.extend(all_intersections)
    return parts, clean_relations, intersections_indices
    
def get_minimal_sym_merge(parts, clean_relations):
    new_parts = []
    existing_labels = set([p['label'] for p in parts])

    compressed_indices = []
    for relation_ind, relation in enumerate(clean_relations):
        indices, params, sym_group_type = relation
        compressed_indices += indices
        if sym_group_type in ["R", "T"]:
            # merge it.
            cur_label = parts[indices[0]]['label'] + "s" # f"_{sym_group_type}_group"
            if cur_label in existing_labels:
                continue
            new_part = dict()
            objs = []
            mesh = o3d.geometry.TriangleMesh()
            for ind in indices:
                mesh += parts[ind]['mesh']
                objs += parts[ind]['objs']
            new_part['objs'] = objs
            new_part['label'] = cur_label
            new_part['original_label'] = parts[indices[0]]['original_label']
            # mesh = mesh.compute_vertex_normals()
            new_part['mesh'] = mesh
            temp_mesh = copy.deepcopy(mesh).scale(SCALE_FACTOR, mesh.get_center())
            pcd = temp_mesh.sample_points_poisson_disk(number_of_points=PC_SIZE)
            new_part['obb'] =  get_oriented_bounding_box_with_fixing(pcd)
            new_part['relation_ind'] = relation_ind
            new_part['children'] = indices
            new_part['hier'] = 0
            new_parts.append(new_part)
            update_hier_order(parts, indices)
            
    return new_parts

def update_hier_order(parts, indices):
    for ind in indices:
        cur_part = parts[ind]
        cur_part['hier'] += 1
        update_hier_order(parts, cur_part['children'])


def cleanup_relations(parts, label_to_sel_syms):

    clean_relations = []
    # Each relation is [index_list, sym_params, sym_type]
    for label, relation_list in label_to_sel_syms.items():
        # The same indice set should be in only one relation
        indice_set_to_divergence = dict()
        indice_to_relation = dict()
        for relation in relation_list:
            c_type = relation[-1]
            clique_list = relation[0]
            if c_type in ["R", "T"]:
                indices, params = get_seq_and_params(clique_list, c_type)
            else:
                clique = relation[0][0]
                # check for flip
                positive_normal = np.array([1, 1, 1])
                positive_normal = positive_normal / np.linalg.norm(positive_normal)
                plane_axis = clique[2]
                if np.dot(positive_normal, plane_axis) < 0:
                    indices = (clique[1], clique[0])
                    params = (-plane_axis, clique[3])
                else:
                    indices = (clique[0], clique[1])
                    params = (plane_axis, clique[3])
            div = get_real_divergence(clique_list, c_type, parts=parts)
            if indices not in indice_set_to_divergence.keys():
                indice_set_to_divergence[indices] = div
                indice_to_relation[indices] = (indices, params, c_type)
            else:
                prev_div = indice_set_to_divergence[indices]
                if div < prev_div:
                    indice_set_to_divergence[indices] = div
                    indice_to_relation[indices] = (indices, params, c_type)
        cur_clean_relations = [v for k, v in indice_to_relation.items()]
        clean_relations.extend(cur_clean_relations)
    return clean_relations

def get_seq_and_params(clique_list, c_type="R"):
    if c_type == "R":
        get_func = get_rotation_clique_to_items
    elif c_type == "T":
        get_func = get_translation_clique_to_items

    selected_relations = dict()
    for item in clique_list:
        # Step 1: Convert to graph
        graph = nx.empty_graph(n=0)
        for item in clique_list:
            ind_1 = item[0]
            ind_2 = item[1]
            if not ind_1 in graph.nodes():
                graph.add_node(ind_1)
            if not ind_2 in graph.nodes():
                graph.add_node(ind_2)
            if not graph.has_edge(ind_1, ind_2):
                graph.add_edge(ind_1, ind_2)
                items = get_func(item)
                for key, value in items.items():
                    graph[ind_1][ind_2][key] = value
            else:
                # replace the edit content if the angle is smaller
                raise ValueError("This should not happen")
    initial_ind = clique_list[0][0]
    ind_seq = [initial_ind] + [successor for successors in dict(nx.bfs_successors(graph, initial_ind)).values() for successor in successors]
    ind_seq = tuple(ind_seq)
    if c_type == "R":
        origin_set = []
        axis_set = []
        for clique in clique_list:
            origin = clique[3]
            origin_set.append(origin)
            axis = clique[2]
            axis_set.append(axis)
        origin = np.mean(origin_set, axis=0)
        axis = np.mean(axis_set, axis=0)
        params = (axis, origin)
    elif c_type == "T":
        delta_set = []
        for clique in clique_list:
            delta = clique[2]
            delta_set.append(delta)
        delta = np.mean(delta_set, axis=0)
        params = (delta,)
    return ind_seq, params

def get_all_sym_relations(parts_with_ids, get_ref_sym = True, get_trans_sym = True, get_rot_sym = True):
    # Step 1: Gather symmetric candidates
    # Labels must match
    syn_candidate_set = defaultdict(list)
    if isinstance(parts_with_ids, list):
        traverse_list = itertools.combinations(parts_with_ids, 2)
    else:
        # its a dict
        traverse_list = itertools.combinations(parts_with_ids.items(), 2)
    for (id_1, part_dict_1), (id_2, part_dict_2) in traverse_list:
            label_1 = part_dict_1['label']
            label_2 = part_dict_2['label']
            if label_1 == label_2:
                syn_candidate_set[label_1].append((id_1, id_2))
    print(f"Number of symmetric candidates: {len(syn_candidate_set)}")
    label_to_syms = defaultdict(list)
    if isinstance(parts_with_ids, list):
        parts = [p for _, p in parts_with_ids]
    else:
        parts = parts_with_ids

    for label, cur_syn_candidate_set in syn_candidate_set.items():
        for syn_candidate in cur_syn_candidate_set:
            ind_1, ind_2 = syn_candidate
            part_dict_1, part_dict_2 = parts[ind_1], parts[ind_2]
            mesh_1 = part_dict_1['mesh']
            mesh_2 = part_dict_2['mesh']

            pcd_1 = mesh_1.sample_points_poisson_disk(number_of_points=PC_SIZE)
            pcd_2 = mesh_2.sample_points_poisson_disk(number_of_points=PC_SIZE)
            # First align 1 to 2 using obb info
            obb_1 = part_dict_1['obb']
            obb_2 = part_dict_2['obb']
            
            # We will create 3 different proposals
            # 1. Reflection proposal
            if get_ref_sym:
                # normal and center.
                center_point = (obb_1.get_center() + obb_2.get_center())/2
                normal = obb_2.get_center() - center_point
                normal_dir = normal / np.linalg.norm(normal)
                transform_mat = get_reflection_matrix(center_point, normal_dir)
                transformation = try_icp(pcd_1, pcd_2, transform_mat)
                if transformation is not None:
                    print(f"--> Reflection Symmetry between {part_dict_1['label']} and {part_dict_2['label']}")
                    sym_set = (ind_1, ind_2, transformation, 0)
                    label_to_syms[label].append(sym_set)
                    sym_set = (ind_1, ind_2, transform_mat, 0)
                    label_to_syms[label].append(sym_set)

            # 2 Translation Candidates
            if get_trans_sym:
                delta = obb_2.get_center() - obb_1.get_center()
                transform_mat = np.eye(4)
                transform_mat[:3, 3] = delta
                transformation = try_icp(pcd_1, pcd_2, transform_mat)
                if transformation is not None:
                    print(f"--> Translation Symmetry between {part_dict_1['label']} and {part_dict_2['label']}")
                    sym_set = (ind_1, ind_2, transformation, 1)
                    label_to_syms[label].append(sym_set)
                    sym_set = (ind_1, ind_2, transform_mat, 1)
                    label_to_syms[label].append(sym_set)
                
            # 3 Rotation Candidates
            # generate all candidates 
            # rotational Sym
            if get_rot_sym:
                rotation_transform_candidates = get_rot_trans_candidates(obb_1, obb_2)
                for ind, transform_mat in enumerate(rotation_transform_candidates):
                    # pcd_transformed = copy.deepcopy(pcd_1).transform(transform_mat)
                    transformation = try_icp(pcd_1, pcd_2, transform_mat)
                    if transformation is not None:
                        print(f"--> Rotation Symmetry between {part_dict_1['label']} and {part_dict_2['label']}")
                        sym_set = (ind_1, ind_2, transformation, 2)
                        # sym_set = (ind_1, ind_2, transform_mat, 2)
                        # I wanna add the conversion to check
                        item = invert_matrix_to_components(np.copy(transformation))
                        if len(item) == 3:
                            label_to_syms[label].append(sym_set)

    # Manual intervention for rotation      
    if get_rot_sym: 
        for label, cur_syn_candidate_set in syn_candidate_set.items():
            # gather all the candidates
            manual_syms = []
            if len(cur_syn_candidate_set) == 1:
                continue
            all_parts = []
            for syn_candidate in cur_syn_candidate_set:
                ind_1, ind_2 = syn_candidate
                all_parts.append(ind_1)
                all_parts.append(ind_2)
            all_parts = list(set(all_parts))
            center_set = []
            for part_id in all_parts:
                center_set.append(parts[part_id]['obb'].get_center())
            # Get center by fitting and
            center_set = np.stack(center_set, 0)
            plane_center = np.mean(center_set, axis=0)
            # Temp
            
            normal_set = []
            for ind_1, ind_2 in zip(all_parts[:-1], all_parts[1:]):
                vec_a = parts[ind_1]['obb'].get_center() - plane_center
                vec_b = parts[ind_2]['obb'].get_center() - plane_center
                vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-16)
                vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-16)

                normal = np.cross(vec_a, vec_b)
                if np.linalg.norm(normal) < 0.1:
                    continue
                normal_dir = normal / (np.linalg.norm(normal) + 1e-16)
                normal_set.append(normal_dir)
            if len(normal_set) == 0:
                continue
            normal = np.mean(normal_set, axis=0)
            normal = normal / (np.linalg.norm(normal) + 1e-16)

            def f_2(c):
                vectors = c[None, :] - center_set
                radii = np.linalg.norm(vectors, axis=1)
                # shift by one
                shifted_radii = np.roll(radii, 1)
                shifted_radii_2 = np.roll(radii, -1)
                ratio_1 = radii / shifted_radii
                ratio_2 = shifted_radii / radii
                delta = 1 - np.minimum(ratio_1, ratio_2)
                ratio_1 = radii / shifted_radii_2
                ratio_2 = shifted_radii_2 / radii
                delta += 1 - np.minimum(ratio_1, ratio_2)
                # print(delta)
                delta += radii
                return delta
            # def f_2(c):
            #     vectors = c[None, :] - center_set
            #     radii = np.linalg.norm(vectors, axis=1)
            #     # shift by one
            #     delta = np.max(radii) - np.min(radii)
            #     delta = np.stack([delta, ] * len(radii), 0)
            #     delta += radii
            #     return delta

            center_2, ier = optimize.leastsq(f_2, plane_center)
            # center = center - np.dot(center - center, normal) * normal
            center = plane_center
            center_2 = center_2 - np.dot(center_2 - plane_center, normal) * normal

            # get amount
            residual_1 = np.mean(f_2(center))
            residual_2 = np.mean(f_2(center_2))
            if residual_1 > 0.1:
                if residual_2 < residual_1:
                    center = center_2

            # get amount
            ind_1 = all_parts[0]
            vec_a = parts[ind_1]['obb'].get_center() - center
            vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-16)
            amount = np.inf
            for ind_2 in all_parts[1:]:
                vec_b = parts[ind_2]['obb'].get_center() - center
                vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-16)
                cosine = np.dot(vec_a, vec_b)
                new_amount = np.arccos(cosine)
                print(new_amount * 180/np.pi)
                if new_amount < amount:
                    amount = new_amount
            temp = o3d.geometry.TriangleMesh()
            rotation = temp.get_rotation_matrix_from_axis_angle(amount * normal)
            # subtract origin
            t1 = np.eye(4)
            t1[:3, 3] = -center
            # multiply by rotation
            t2 = np.eye(4)
            t2[:3, :3] = rotation
            # add origin
            t3 = np.eye(4)
            t3[:3, 3] = center
            transform_mat = t3 @ t2 @ t1
            for syn_candidate in cur_syn_candidate_set:
                ind_1, ind_2 = syn_candidate
                part_dict_1, part_dict_2 = parts[ind_1], parts[ind_2]
                mesh_1 = part_dict_1['mesh']
                mesh_2 = part_dict_2['mesh']
                obb_1 = part_dict_1['obb']
                obb_2 = part_dict_2['obb']
                box_mesh_1 = obb_to_mesh(obb_1)
                box_mesh_2 = obb_to_mesh(obb_2)
                pcd_1 = box_mesh_1.sample_points_poisson_disk(number_of_points=PC_SIZE)
                pcd_2 = box_mesh_2.sample_points_poisson_disk(number_of_points=PC_SIZE)

                transformed_pcd_1 = copy.deepcopy(pcd_1).transform(transform_mat)
                h_dist = hausdorff_distance(np.asarray(transformed_pcd_1.points),
                                            np.asarray(pcd_2.points))
                if h_dist < ROT_HAUS_DIST_THRESHOLD:
                    pcd_1 = mesh_1.sample_points_poisson_disk(number_of_points=PC_SIZE)
                    pcd_2 = mesh_2.sample_points_poisson_disk(number_of_points=PC_SIZE)
                    transformation_2 = try_icp(pcd_1, pcd_2, transform_mat)
                    if transformation_2 is not None:
                        print(f"--> Manual Rotation Symmetry between {part_dict_1['label']} and {part_dict_2['label']}")
                        sym_set = (ind_1, ind_2, np.copy(transform_mat), 2)
                        manual_syms.append(sym_set)
            # should try translation sym as well:
            # if is_colinear(center_set):
            # TODO: Add translation candidates as well.

        
            label_to_syms[label] = manual_syms + label_to_syms[label]
        print("added Manual Rotations")
        
    return label_to_syms


def prune_sym_relations(label_to_syms, parts_with_ids, 
                        ref_axis_aligned = True, 
                        test_subset_of_failed_relations=True,
                        allow_overlap=False,
                        min_size_relation_group=3,):
    if isinstance(parts_with_ids, list):
        parts = [p for _, p in parts_with_ids]
    else:
        parts = parts_with_ids
    label_to_sel_syms = defaultdict(list)
    for label, cur_selected_syms in label_to_syms.items():
        # Step 1: Gather symmetries based on types
        reflection_cliques, rot_cliques, translation_cliques = get_all_cliques(cur_selected_syms, parts)
        # step 2: From cliques get parametric relations
        rotation_relation_cliques = get_pruned_via_graph(rot_cliques, "R", min_size_relation_group)
        translation_relation_cliques = get_pruned_via_graph(translation_cliques, "T", min_size_relation_group)
        
        # Reject/subset based on divergence.
        all_cliques = []

        for clique_list in rotation_relation_cliques.values():
            item = (clique_list, "R")
            all_cliques.append(item)
        for clique_list in translation_relation_cliques.values():
            item = (clique_list, "T")
            all_cliques.append(item)

        div = [get_real_divergence(*x, parts) for x in all_cliques]
        print("got divs")
        pruned_cliques = []
        for ind, clique in enumerate(all_cliques):
            if div[ind] < DIV_THRESHOLD:
                pruned_cliques.append(clique)
            else:
                # select a subset:
                if test_subset_of_failed_relations:
                    clique_dict = {0: clique[0]}
                    c_type= clique[1]
                    if c_type == "T":
                        clique = gather_divergence_based_subset(*clique, parts)
                        # Clean it
                        clique_dict = {0: clique[0]}
                        clique = get_pruned_via_graph(clique_dict, c_type, min_size_relation_group)
                        clique_set, _, _ = extract_selected_relations(clique, c_type=c_type)
                        for clique in clique_set:
                            if len(clique[0]) > 1:
                                pruned_cliques.append(clique)
        # Now reject subsets
        all_cliques = pruned_cliques
        # Reject based on subsetting.
        # selected_cliques = [x for ind, x in enumerate(selected_cliques) if div[ind] < DIV_THRESHOLD]
        pruned_rot_cliques = dict()
        count = 0
        for clique in all_cliques:
            if clique[-1] == "R":
                pruned_rot_cliques[count] = clique[0]
                count += 1
        pruned_translation_cliques = dict()
        count = 0
        for clique in all_cliques:
            if clique[-1] == "T":
                pruned_translation_cliques[count] = clique[0]
                count += 1

        # First from cliques get relations using 
        
        selected_rot_cliques, _, _ = extract_selected_relations(pruned_rot_cliques, c_type="R")
        selected_translation_cliques, _, _ = extract_selected_relations(pruned_translation_cliques, c_type="T")
        selected_rot_cliques = [x for x in selected_rot_cliques if x is not None]
        selected_translation_cliques = [x for x in selected_translation_cliques if x is not None]
        
        selected_cliques = selected_translation_cliques + selected_rot_cliques
        # reflections -> only allow axis aligned - maybe not.
        clique_ind_to_clique_div = dict()
        for ind, clique in enumerate(selected_cliques):
            clique_id = get_clique_id(clique[0])
            cur_div = get_real_divergence(clique[0], clique[-1], parts)
            if clique_id in clique_ind_to_clique_div.keys():
                if cur_div < clique_ind_to_clique_div[clique_id][1]:
                    clique_ind_to_clique_div[clique_id] = (clique, cur_div)
            else:
                clique_ind_to_clique_div[clique_id] = (clique, cur_div)
        # sort the keys by size
        clique_inds = sorted(clique_ind_to_clique_div.keys(), key=lambda x: -len(x))
        pruned_cliques = dict()
        for clique_ind_key in clique_inds:
            clique, cur_div = clique_ind_to_clique_div[clique_ind_key]
            is_subset = False
            is_overlapping = False
            for key in pruned_cliques.keys():
                if all([x in key for x in clique_ind_key]):
                    is_subset = True
                    break
                if any([x in key for x in clique_ind_key]):
                    is_overlapping = True
                    # break
            if not is_subset:
                if allow_overlap:
                    pruned_cliques[clique_ind_key] = clique
                else:
                    if not is_overlapping:
                        pruned_cliques[clique_ind_key] = clique
            
        selected_cliques = [x for _, x in pruned_cliques.items()]
        clique_to_inds = []
        for sel_clique in selected_cliques:
            clique_list, _ = sel_clique
            indices = []
            for clique in clique_list:
                index_1 = clique[0]
                index_2 = clique[1]
                indices.append(index_1)
                indices.append(index_2)
            indices = list(set(indices))
            clique_to_inds.append(indices)
        sel_reflection_cliques = []
        for ind, ref in reflection_cliques.items():
            index_1, index_2, plane_normal, plane_origin, = ref[0]
            if ref_axis_aligned:
                # get normal of the reflection

                # check if normal is axis-aligned
                cos_x = np.dot(plane_normal, np.array([1, 0, 0]))
                cos_y = np.dot(plane_normal, np.array([0, 1, 0]))
                cos_z = np.dot(plane_normal, np.array([0, 0, 1]))
                max_cos = np.max([np.abs(x) for x in [cos_x, cos_y, cos_z]])
                if max_cos > REF_AXIS_ALIGNMENT_THRESHOLD:
                    feasible = True
                else:
                    feasible = False
            else:
                feasible = True
            if feasible:
                is_subset_clique = False
                is_overlap_clique = False
                for clique_inds in clique_to_inds:
                    if (index_1 in clique_inds) and (index_2 in clique_inds):
                        is_subset_clique = True
                        # Temporary hack - add all relations
                        break
                    elif (index_1 in clique_inds) or (index_2 in clique_inds):
                        is_overlap_clique = True

                if not is_subset_clique:
                    if allow_overlap:
                        sel_reflection_cliques.append(ref)
                    else:
                        if not is_overlap_clique:
                            sel_reflection_cliques.append(ref)
        # Add a 
        ref_cliques = [(clique_list, "REF") for clique_list in sel_reflection_cliques]
        div = [get_real_divergence(*x, parts) for x in ref_cliques]
        print("got divs")
        ref_cliques = [x for ind, x in enumerate(ref_cliques) if div[ind] < REF_DIV_THRESHOLD]
        clique_to_div = dict()
        selected_ref_cliques = dict()
        for ind, clique in enumerate(ref_cliques):
            clique_id = get_clique_id(clique[0])
            cur_div = div[ind]
            if clique_id in clique_to_div.keys():
                if cur_div < clique_to_div[clique_id]:
                    clique_to_div[clique_id] = cur_div
                    selected_ref_cliques[clique_id] = clique
            else:
                clique_to_div[clique_id] = cur_div
                selected_ref_cliques[clique_id] = clique

        ref_cliques = [y for x, y in selected_ref_cliques.items()]
        selected_cliques.extend(ref_cliques)
        label_to_sel_syms[label] = selected_cliques
    return label_to_sel_syms

def extract_selected_relations(rot_cliques, c_type="R", 
                               clique_id_to_index=None, 
                               clique_id_to_divergence=None,
                               selected_cliques=None):

    if clique_id_to_divergence is None:
        clique_id_to_index = dict()
        clique_id_to_divergence = dict()
        selected_cliques = []
    for index, clique_list in rot_cliques.items():
        if len(clique_list) >= (MIN_SIZE_RELATION_GROUP - 1):
            clique_id = get_clique_id(clique_list)
                # sub-cliques - not allowed.
            is_subset = False
            match_clique = False
            for prev_clique_id, prev_index in clique_id_to_index.items():
                if all([x in prev_clique_id for x in clique_id]):
                    if len(clique_id) == len(prev_clique_id):
                        match_clique = True
                    else:
                        is_subset = True
                    break
            if is_subset:
                    # Do not take subsets
                continue
            else:
                if match_clique:
                        # update based on divergence
                    old_divergence = clique_id_to_divergence[clique_id]
                    clique_index = clique_id_to_index[clique_id]
                    new_divergence = get_divergence(clique_list, c_type)
                    if new_divergence < old_divergence:
                        selected_cliques[clique_index] = (clique_list, c_type)
                        clique_id_to_divergence[clique_id] = new_divergence
                else:
                    # Must add this one
                    selected_cliques.append((clique_list, c_type))
                    clique_id_to_index[clique_id] = len(selected_cliques) - 1
                    clique_id_to_divergence[clique_id] = get_divergence(clique_list, c_type)
                    # should now go through all the cliques and check if they are subsets
                    index_to_pop = []
                    id_to_pop = []
                    for prev_clique_id, prev_index in clique_id_to_index.items():
                        if all([x in clique_id for x in prev_clique_id]):
                            if len(prev_clique_id) != len(clique_id):
                                # remove the previous one
                                index_to_pop.append(prev_index)
                                id_to_pop.append(prev_clique_id)
                    for prev_index in sorted(index_to_pop, reverse=True):
                        if prev_index < len(selected_cliques):
                            selected_cliques[prev_index] = None
                    for prev_id in id_to_pop:
                        del clique_id_to_index[prev_id]
                        del clique_id_to_divergence[prev_id]

    # selected_cliques = [x for x in selected_cliques if x is not None]
    return selected_cliques, clique_id_to_index, clique_id_to_divergence


def get_rotation_clique_to_items(clique):

    axis = clique[2]
    origin = clique[3]
    angle = np.linalg.norm(axis)
    axis = axis / angle
    items = {'axis': axis, 'origin': origin, 'angle': angle}
    return items

def get_translation_clique_to_items(clique):
    delta = clique[2]
    amount = np.linalg.norm(delta)
    items = {'delta': delta, 'amount': amount}
    return items


def get_pruned_via_graph(cliques_dict, c_type="R", min_size_relation_group=MIN_SIZE_RELATION_GROUP):
    if c_type == "R":
        get_func = get_rotation_clique_to_items
        edge_weight = "angle"
    elif c_type == "T":
        get_func = get_translation_clique_to_items
        edge_weight = "amount"

    selected_relations = dict()
    for ind, clique_list in cliques_dict.items():
        # Step 1: Convert to graph
        if len(clique_list) >= (min_size_relation_group - 1):
            # graph = nx.empty_graph(n=0)
            graph = nx.DiGraph()
            for item in clique_list:
                ind_1 = item[0]
                ind_2 = item[1]
                if not ind_1 in graph.nodes():
                    graph.add_node(ind_1)
                if not ind_2 in graph.nodes():
                    graph.add_node(ind_2)
                if graph.has_edge(ind_1, ind_2):
                    items = get_func(item)
                    value = items[edge_weight]
                    if value < graph[ind_1][ind_2][edge_weight]:
                        for key, value in items.items():
                            graph[ind_1][ind_2][key] = value
                else:
                    graph.add_edge(ind_1, ind_2)
                    items = get_func(item)
                    for key, value in items.items():
                        graph[ind_1][ind_2][key] = value
                    # replace the edit content if the angle is smaller
                # Can happen when rotation is by 180 degrees
                # raise ValueError("This should not happen")
            # step 2: For each connected component generate spanning tree as relation.
            # for subgraph_nodes in nx.weakly_connected_components(graph):
            for subgraph_nodes in semiconnected_components(graph):
                if len(subgraph_nodes) >= min_size_relation_group:
                    subgraph = graph.subgraph(subgraph_nodes).copy()
                    ind_seq = subgraph_nodes
                    new_relation = []
                    for ind_1, ind_2 in zip(ind_seq[:-1], ind_seq[1:]):
                        edge_data = subgraph.get_edge_data(ind_1, ind_2)
                        if c_type == "R":
                            axis = edge_data["axis"]
                            origin = edge_data["origin"]
                            angle = edge_data["angle"]
                            new_relation.append((ind_1, ind_2, axis * angle, origin))
                        elif c_type == "T":
                            delta = edge_data["delta"]
                            new_relation.append((ind_1, ind_2, delta))
                    selected_relations[len(selected_relations.keys())] = new_relation
    return selected_relations

def semiconnected_components(G):
    # https://stackoverflow.com/questions/76475479/is-there-a-way-to-find-semi-connected-unilaterally-connected-components-in-a
    G0 = nx.quotient_graph(G, list(nx.strongly_connected_components(G)))
    G1 = nx.transitive_reduction(G0)
    sources = {v for v in G1.nodes() if G1.in_degree(v) == 0}
    sinks = {v for v in G1.nodes() if G1.out_degree(v) == 0}
    # map the frozen set to a seq
    for source in sources:
        for path in nx.all_simple_paths(G1, source, sinks):
            full_path = []
            last_node_index = None
            first = path[0]
            last = path[-1]
            start_indices_to_try = []
            if len(first) > 1:
                start_indices_to_try.extend(list(first))
            else:
                start_indices_to_try.append(list(first)[0])
            end_indices_to_try = []
            if len(last) > 1:
                end_indices_to_try.extend(list(last))
            else:
                end_indices_to_try.append(list(last)[0])
            for start, end in itertools.product(start_indices_to_try, end_indices_to_try):
                for underlying_path in nx.all_simple_paths(G, start, end):
                    yield underlying_path
            # yield list(itertools.chain(*path))
    # Work around a bug in all_simple_paths (does not return length-0 paths, see
    # https://github.com/networkx/networkx/issues/6690).
    intersecting = sources & sinks
    for node_set in intersecting:
        node_set = list(node_set)
        subgraph = G.subgraph(node_set).copy()
        mst = nx.minimum_spanning_tree(subgraph.to_undirected())
        directed_version = nx.DiGraph(mst)
        remove_edges = []
        for edge in directed_version.edges():
            if directed_version.has_edge(*edge[::-1]):
                if not edge[::-1] in remove_edges:
                    remove_edges.append(edge)
        for edge in remove_edges:
            directed_version.remove_edge(*edge)
        edges_to_flip = []
        for edge in directed_version.edges():
            # if edge is not in G flip it:
            if not G.has_edge(*edge):
                edges_to_flip.append(edge)
        for edge in edges_to_flip:
            directed_version.remove_edge(*edge)
            directed_version.add_edge(*edge[::-1])
                
        sources = {v for v in directed_version.nodes() if directed_version.in_degree(v) == 0}
        sinks = {v for v in directed_version.nodes() if directed_version.out_degree(v) == 0}
        for source in sources:
            for path in nx.all_simple_paths(directed_version, source, sinks):
                yield path
    # yield from map(list, sources & sinks)


def get_all_cliques(cur_selected_syms, parts=None):
    all_reflections = []
    all_translations = []
    all_rots = []
    for selected_sym in cur_selected_syms:
        ind_1, ind_2, transformation, transform_type = selected_sym
        if transform_type == 0:
            try:
                t_type, plane_normal, plane_origin = invert_matrix_to_components(transformation)
            except:
                print("Failed to invert reflection matrix")
                t_type = "NIL"
            if t_type == "REF":
                all_reflections.append((ind_1, ind_2, plane_normal, plane_origin))
        elif transform_type == 1:
                # translation -> no rotation
            try:
                t_type, translation = invert_matrix_to_components(transformation)
                amount = np.linalg.norm(translation)
            except:
                print("Failed to invert translation matrix")
                amount = 0
            if amount > TRANS_AMOUNT_THRESHOLD:
                print("viable translation")
                all_translations.append((ind_1, ind_2, translation))
        elif transform_type == 2:
                # rotation:
            try:
                t_type, axis, origin = invert_matrix_to_components(transformation)
                # map angle into the range -np.pi to np.pi
                angle = np.linalg.norm(axis)
                angle = (angle + np.pi) % (2*np.pi) - np.pi
                axis = axis / np.linalg.norm(axis) * angle
            except:
                print("Failed to invert rotation matrix")
                angle = 0
            if np.abs(angle) > ROT_AMOUNT_THRESHOLD:
                if np.abs(angle) < (5 * np.pi / 6):
                    print("viable rotation")
                    all_rots.append((ind_1, ind_2, axis, origin))

    # Step 2: Gather equivalent rotations relations in cliques
    rot_cliques = defaultdict(list)
    check_set = []
    for rot in all_rots:
        index_1, index_2, axis, origin = rot
            # when are two rotations equivalent - when the axis match and the translation is also similar
        if len(check_set) == 0:
            check_set.append([index_1, index_2, axis, origin])
            rot_cliques[0].append(rot)
        else:
            found_match = False
            for ind, (prev_1, prev_2, prev_axis, prev_origin) in enumerate(check_set):
                    # check if the rotation is similar
                delta_1 = np.linalg.norm(prev_axis - axis)
                delta_2 = np.linalg.norm(prev_axis + axis)
                if line_close(prev_axis, axis, prev_origin, origin):
                    print("Found_match")
                    if delta_1 < 0.075:
                        rot_cliques[ind].append(rot)
                        found_match = True
                        # break
                    elif delta_2 < 0.075:
                        inverted_rot = (index_2, index_1, -axis, origin)
                        rot_cliques[ind].append(inverted_rot)
                        found_match = True
            if not found_match:
                check_set.append([index_1, index_2, axis, origin])
                rot_cliques[len(check_set)-1].append(rot)
        # same for translations
    translation_cliques = defaultdict(list)
    check_set = []
    for trans in all_translations:
        index_1, index_2, translation_vec = trans
        if len(check_set) == 0:
            origin = parts[index_1]['obb'].get_center()
            check_set.append((origin, translation_vec))
            translation_cliques[0].append(trans)
        else:
            found_match = False
            for ind, (prev_origin, prev_trans_vec) in enumerate(check_set):
                # This should be a line close

                # if axis_close(prev_trans_vec, translation_vec):
                if axis_close(prev_trans_vec, translation_vec, prev_origin, origin):
                    delta_1 = np.linalg.norm(prev_trans_vec - translation_vec)
                    if delta_1 < 0.075:
                        translation_cliques[ind].append(trans)
                        found_match = True
                        # break
                if axis_close(prev_trans_vec, -translation_vec, prev_origin, origin):
                    delta_1 = np.linalg.norm(prev_trans_vec + translation_vec)
                    if delta_1 < 0.075:
                        inverted_trans = (index_2, index_1, -translation_vec)
                        translation_cliques[ind].append(inverted_trans)
                        found_match = True
                        # break
            if not found_match:
                check_set.append((origin, translation_vec))
                translation_cliques[len(check_set)-1].append(trans)
    # Reflection cliques
    # All reflections are independant
    reflection_cliques = dict()
    for ind, ref in enumerate(all_reflections):
        reflection_cliques[ind] = [ref]
    return reflection_cliques, rot_cliques, translation_cliques

def get_clique_id(clique_list):
    indices = []
    for clique in clique_list:
        index_1 = clique[0]
        index_2 = clique[1]
        indices.append(index_1)
        indices.append(index_2)
    indices.sort()
    indices = tuple(set(indices))
    # clique_id = "_".join([str(x) for x in indices])
    return indices

def get_divergence(prev_clique_set, c_type="R"):
    divergence = 0
    for transform_1, transfrom_2 in itertools.combinations(prev_clique_set, 2):
        if c_type == "R":
            axis_1, origin_1 = transform_1[2], transform_1[3]
            axis_2, origin_2 = transfrom_2[2], transfrom_2[3]
            angle_1 = np.linalg.norm(axis_1)
            angle_2 = np.linalg.norm(axis_2)
            axis_1 = axis_1 / angle_1
            axis_2 = axis_2 / angle_2
            diff = np.linalg.norm(np.cross(axis_1, axis_2))
            diff += np.linalg.norm(origin_1 - origin_2)
            diff += np.abs(angle_1 - angle_2)
        elif c_type == "T":
            delta_1 = transform_1[2]
            delta_2 = transfrom_2[2]
            amount_1 = np.linalg.norm(delta_1)
            amount_2 = np.linalg.norm(delta_2)
            delta_1 = delta_1 / amount_1
            delta_2 = delta_2 / amount_2
            diff = np.linalg.norm(np.cross(delta_1, delta_2))
            diff += np.abs(amount_1 - amount_2)
        elif c_type == "REF":
            raise NotImplementedError
        divergence += diff
    n_items = len(prev_clique_set)
    divergence = divergence / max(1, (n_items * (n_items - 1) / 2))
    return divergence

def get_real_divergence(clique_set, c_type="R", parts=None, return_list=False):
    divergence = []
    transform_mat = None
    if c_type == "R":
        origin_set = []
        axis_set = []
        for clique in clique_set:
            origin = clique[3]
            origin_set.append(origin)
            axis = clique[2]
            axis_set.append(axis)
        origin = np.mean(origin_set, axis=0)
        axis = np.mean(axis_set, axis=0)
    elif c_type == "T":
        delta_set = []
        for clique in clique_set:
            delta = clique[2]
            delta_set.append(delta)
        delta = np.mean(delta_set, axis=0)

    for clique in clique_set:
        # create the average transform
        index_1, index_2 = clique[0], clique[1]
        if transform_mat is None:
            if c_type == "R":
                t_1 = np.eye(4)
                t_1[:3, 3] = -origin
                t_2 = np.eye(4)
                axis = th.from_numpy(axis)
                t_2[:3, :3] = pytorch3d.transforms.axis_angle_to_matrix(axis).cpu().numpy()
                t_3 = np.eye(4)
                t_3[:3, 3] = origin
                transform_mat = t_3 @ t_2 @ t_1
            elif c_type == "T":
                transform_mat = np.eye(4)
                transform_mat[:3, 3] = delta
            elif c_type == "REF":
                plane_normal, plane_origin = clique[2], clique[3]
                transform_mat = get_reflection_matrix(plane_origin, plane_normal)


        mesh_1 = parts[index_1]['mesh']
        mesh_2 = parts[index_2]['mesh']
        # obb_1 = parts[index_1]['obb']
        # obb_2 = parts[index_2]['obb']
        pcd_1 = mesh_1.sample_points_poisson_disk(number_of_points=PC_SIZE)
        pcd_2 = mesh_2.sample_points_poisson_disk(number_of_points=PC_SIZE)
        transformed_pcd_1 = copy.deepcopy(pcd_1).transform(transform_mat)
        diff = hausdorff_distance(np.asarray(transformed_pcd_1.points),
                                    np.asarray(pcd_2.points))
        divergence.append(diff)
    # divergence = np.mean(divergence)
    if return_list:
        return divergence
    else:
        divergence = np.max(divergence)
        return divergence

def gather_divergence_based_subset(clique_set, c_type="R", parts=None):
    divergence = get_real_divergence(clique_set, c_type, parts, return_list=True)
    # divergence = np.mean(divergence)
    selected_syms = []
    for i in range(len(divergence)):
        if divergence[i] < DIV_THRESHOLD:
            selected_syms.append(clique_set[i])
    return_format = (selected_syms, c_type)
    return return_format

def get_intersections(parts_with_ids, label_to_sel_syms, other_parts_with_ids=None):

    clique_to_inds = []
    for label, selected_cliques in label_to_sel_syms.items():
        for sel_clique in selected_cliques:
            clique_list, _ = sel_clique
            indices = []
            for clique in clique_list:
                index_1 = clique[0]
                index_2 = clique[1]
                indices.append(index_1)
                indices.append(index_2)
            indices = list(set(indices))
            clique_to_inds.append(indices)

    overlaps = []
    if other_parts_with_ids is None:
        all_items = itertools.combinations(parts_with_ids, 2)
        other_parts_with_ids = parts_with_ids
    else:
        all_items = itertools.product(parts_with_ids, other_parts_with_ids)
        if len(parts_with_ids) > 1:
            additional = itertools.combinations(parts_with_ids, 2)
            all_items = itertools.chain(all_items, additional)
    for (id_1, part_dict_1), (id_2, part_dict_2) in all_items:
        obb_1 = part_dict_1['obb']
        obb_2 = part_dict_2['obb']
        # Get points cloud of 1
        index_list_1, index_list_2, index_list, sel_points = extract_obb_intersection_pts(obb_1, obb_2)

        if len(index_list) > 0:
            inter_pcd = o3d.geometry.PointCloud()
            inter_pcd.points = o3d.utility.Vector3dVector(sel_points)
            inter_obb = get_oriented_bounding_box_with_fixing(inter_pcd)
            volume = inter_obb.volume()
            # Robustness control
            if volume >= MIN_VOLUME:
                # only if they are not in the same clique
                within_clique = False
                for clique_inds in clique_to_inds:
                    if (id_1 in clique_inds) and (id_2 in clique_inds):
                        within_clique = True
                        break
                if not within_clique:
                    print(f"Intersection between {part_dict_1['label']} and {part_dict_2['label']}")
                    # analyze the contact - is it point, line or face contact?
                    order_1 = get_self_sym_order(obb_2, obb_1)
                    order_2 = get_self_sym_order(obb_1, obb_2)
                    order = min(order_1, order_2)
                    n_points = np.asarray(inter_pcd.points).shape[0]
                    # if len(index_list_1) == n_points:
                    # #     # pcd_1 is inside pcd_2
                    #     print("pcd_1 is inside pcd_2")
                    #     order = 3
                    # elif len(index_list_2) == n_points:
                    # #     # pcd_2 is inside pcd_1
                    #     print("pcd_2 is inside pcd_1")
                    #     order = 3
                    overlaps.append((id_1, id_2, sel_points, order))
    return overlaps

def update_intersection_by_relations(input_intersections, clean_relation_set):
    for relation in clean_relation_set:
        indices, params, c_type = relation
        # if all of them contact some other object
        selected_intersections = []
        for intersection in input_intersections:
            id_1, id_2, sel_points, order = intersection
            if id_1 in indices or id_2 in indices:
                selected_intersections.append(intersection)
            elif id_1 in indices and id_2 in indices:
                print("WUT?")
        grouped_by_other = defaultdict(list)
        for intersection in selected_intersections:
            id_1, id_2, sel_points, order = intersection
            if id_1 in indices:
                grouped_by_other[id_2].append(intersection)
            elif id_2 in indices:
                grouped_by_other[id_1].append(intersection)
        for other_id, intersections in grouped_by_other.items():
            min_order = np.inf
            for intersection in intersections:
                id_1, id_2, sel_points, order = intersection
                if order < min_order:
                    min_order = order
            for intersection in intersections:
                id_1, id_2, sel_points, order = intersection
                index = input_intersections.index(intersection)
                if min_order != intersection[-1]:
                    print("Corrected order!")
                input_intersections[index] = (id_1, id_2, sel_points, min_order)
    return input_intersections

def extract_obb_intersection_pts(obb_1, obb_2):

    # pcd = obb.TriangleMesh.sample_points_uniformly(number_of_points=PC_SIZE)
    pcd_1 = obb_to_mesh(obb_1).sample_points_poisson_disk(number_of_points=PC_SIZE)
    index_list_1 = obb_2.get_point_indices_within_bounding_box(pcd_1.points)
    
    pcd_2 = obb_to_mesh(obb_2).sample_points_poisson_disk(number_of_points=PC_SIZE)
    index_list_2 = obb_1.get_point_indices_within_bounding_box(pcd_2.points)
    
    index_list = index_list_1 + index_list_2

    if len(index_list) > 0:
        points_1 = np.asarray(pcd_1.points)
        points_2 = np.asarray(pcd_2.points)
        sel_points_1 = points_1[index_list_1]
        sel_points_2 = points_2[index_list_2]
        sel_points = np.concatenate([sel_points_1, sel_points_2], axis=0)
    else:
        sel_points = []
    return index_list_1, index_list_2, index_list, sel_points

def get_self_sym_order(obb_1, obb_2):
    
    points = np.asarray(obb_2.get_box_points())
    obb_2_pcd = o3d.geometry.PointCloud()
    obb_2_pcd.points = o3d.utility.Vector3dVector(points)
    # Convert to bbox?

    center = obb_1.get_center()
    normals, sizes = extract_normals_and_sizes(obb_1)
    order = 0
    for ind, cur_nor in enumerate(normals):
        transform_mat = get_reflection_matrix(center, cur_nor)
        transfromed_points = copy.deepcopy(obb_2_pcd).transform(transform_mat)
        h_dist = hausdorff_distance(np.asarray(transfromed_points.points),
                                    np.asarray(obb_2_pcd.points))
        
        if h_dist < HAUS_DIST_TRHESHOLD:
            if h_dist < sizes[ind] * 0.5:
                order += 1
    return order

# TODO: Consider a pcd dependant version
# THe amount of delta allowed should depend on the size of the pcds.
def try_icp(pcd_1, pcd_2, transformation):
    final_transform = None
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_1, pcd_2, ICP_THRESHOLD, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ICP_ITER))
    # if reg_p2p.fitness > ICP_FITNESS_THRESHOLD:
    transformation = reg_p2p.transformation
    transformed_pcd_1 = copy.deepcopy(pcd_1).transform(transformation)        
    h_dist = hausdorff_distance(np.asarray(transformed_pcd_1.points),
                        np.asarray(pcd_2.points))
    print("h_dist", h_dist)
    if h_dist < HAUS_DIST_TRHESHOLD:
        final_transform = transformation
    return final_transform
