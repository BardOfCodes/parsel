import numpy as np
from collections import defaultdict
from pathlib import Path
import open3d as o3d

direction_array = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],])

direction_with_edge_center_array = np.array([
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 1],
    [1, -1, 0],
    [0, -1, -1],
    [-1, -1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, -1],
    [-1, 1, 0],
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1]])
direction_with_edge_center_array = direction_with_edge_center_array / np.linalg.norm(direction_with_edge_center_array, axis=1, keepdims=True)

rotation_map = {
    (0, 4): "down",
    (1, 4): "up",
    (2, 4): "front",
    (3, 4): "back",
    (4, 4): "left",
    (5, 4): "right",
    (6, 4): "front_down",
    (7, 4): "right_down",
    (8, 4): "back_down",
    (9, 4): "left_down",
    (10, 4): "front_up",
    (11, 4): "right_up",
    (12, 4): "back_up",
    (13, 4): "left_up",
    (14, 4): "front_left",
    (15, 4): "front_right",
    (16, 4): "back_right",
    (17, 4): "back_left", 
    # final situation
    (0, 8):"top_front_right",
    (1, 8): "top_front_left",
    (2, 8): "top_back_left",
    (3, 8): "top_back_right",
    (4, 8): "bottom_front_right",
    (5, 8): "bottom_front_left",
    (6, 8): "bottom_back_left",
    (7, 8): "bottom_back_right"
}
direction_map = {
    0: "right",
    1: "left",
    2: "up",
    3: "down",
    4: "front",
    5: "back",
}
inverse_direction = {
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
    "front": "back",
    "back": "front",
}
translation_map = {
    (0, 3): ["left", "center", "right"],
    (1, 3): ["right", "center", "left"],
    (2, 3): ["down", "center", "up"],
    (3, 3): ["up", "center", "down"],
    (4, 3): ["back", "center", "front"],
    (5, 3): ["front", "center", "back"],
    (0, 4): ["left", "center_left", "center_right", "right"],
    (1, 4): ["right", "center_right", "center_left", "left"],
    (2, 4): ["down", "center_down", "center_up", "up"],
    (3, 4): ["up", "center_up", "center_down", "down"],
    (4, 4): ["back", "center_back", "center_front", "front"],
    (5, 4): ["front", "center_front", "center_back", "back"],
    (0, 5): ["left", "center_left", "center", "center_right", "right"],
    (1, 5): ["right", "center_right", "center", "center_left", "left"],
    (2, 5): ["down", "center_down", "center", "center_up", "up"],
    (3, 5): ["up", "center_up", "center", "center_down", "down"],
    (4, 5): ["back", "center_back", "center", "center_front", "front"],
    (5, 5): ["front", "center_front", "center", "center_back", "back"],
}
# group by hier

def relabel_parts(parts_graph, clean_relations):
    stack = [0]
    while(len(stack) > 0):
        cur_id = stack.pop()
        cur_node = parts_graph.nodes[cur_id]
        # get all successors
        successors = list(parts_graph.successors(cur_id))
        label_to_node_ids = defaultdict(list)
        for successor in successors:
            label = parts_graph.nodes[successor]['part']['label']
            label_to_node_ids[label].append(successor)
        for label, node_ids in label_to_node_ids.items():
            mode = "R"
            if len(node_ids) == 1:
                continue
            if len(node_ids) == 2:
                # ref relabeling.
                mode = "REF"
            else:
                if len(node_ids) <=5:
                    part_1 = parts_graph.nodes[node_ids[0]]['part']
                    part_2 = parts_graph.nodes[node_ids[1]]['part']
                    obb_1 = part_1['obb']
                    obb_2 = part_2['obb']
                    center_1 = obb_1.get_center()
                    center_2 = obb_2.get_center()
                    norm = np.inf
                    sel_ind = 0
                    for next_ind in node_ids[1:]:
                        temp_center_2 = parts_graph.nodes[next_ind]['part']['obb'].get_center()
                        temp = temp_center_2 - center_1
                        if np.linalg.norm(temp) < norm:
                            center_2 = parts_graph.nodes[next_ind]['part']['obb'].get_center()
                            norm = np.linalg.norm(temp)
                            delta_1 = temp
                            sel_ind = next_ind
                    delta_1 = delta_1 / np.linalg.norm(delta_1)
                    dot_prod = -np.inf
                    best_ind = 0
                    for next_ind in node_ids[1:]:
                        if next_ind == sel_ind:
                            continue
                        part_3 = parts_graph.nodes[next_ind]['part']
                        obb_3 = part_3['obb']
                        center_3 = obb_3.get_center()
                        delta_2 = center_3 - center_2
                        delta_2 = delta_2 / np.linalg.norm(delta_2)
                    # now they should around 60*
                        cur_dot_prod = np.dot(delta_1, delta_2)
                        if cur_dot_prod > dot_prod:
                            dot_prod = cur_dot_prod
                            best_ind = next_ind
                    # TODO: Fix this
                    
                    angle_delta = np.pi/4
                    if len(node_ids) == 3:
                        # angle around np.pi/3
                        min_bound = np.cos(2 * np.pi/3 + angle_delta)
                        max_bound = np.cos(2 * np.pi/3 - angle_delta)
                    elif len(node_ids) == 4:
                        # rot or translation.
                        # should be around 90*
                        min_bound = np.cos(np.pi/2 + angle_delta)
                        max_bound = np.cos(np.pi/2 - angle_delta)
                    elif len(node_ids) == 5:
                        min_bound = np.cos(3 * np.pi/5 + angle_delta)
                        max_bound = np.cos(3 * np.pi/5 - angle_delta)

                    if min_bound < dot_prod < max_bound:
                        # rot
                        mode = "R"
                    else:
                        # translation
                        mode = "T"

                    # centers = [parts_graph.nodes[node_id]['part']['obb'].get_center() for node_id in node_ids]
                    # centers = np.array(centers)
                    # center = np.mean(centers, axis=0)
                    # dirs = centers - center
                    # dir_sum = np.sum(dirs, axis=0)
                    # norm_sum = np.linalg.norm(dir_sum)
                    # if norm_sum < 0.1:
                    #     mode = "R"
                    # else:
                    #     mode = "T"

            if mode == "REF":
                part_id_1, part_id_2 = node_ids
                part_1 = parts_graph.nodes[part_id_1]['part']
                part_2 = parts_graph.nodes[part_id_2]['part']
                obb_1 = part_1['obb']
                obb_2 = part_2['obb']
                center_1 = obb_1.get_center()
                center_2 = obb_2.get_center()
                delta = center_2 - center_1
                delta = delta / np.linalg.norm(delta)
                dot_prods = (direction_array * delta[None, :]).sum(axis=1)
                max_ind = np.argmax(dot_prods)
                direction = direction_map[max_ind]
                part_2['label'] = f"{label}_{direction}"
                part_1['label'] = f"{label}_{inverse_direction[direction]}"
            elif mode == "R":
                if len(node_ids) == 4:
                    # consider naming based on angle to points
                    centers = [parts_graph.nodes[node_id]['part']['obb'].get_center() for node_id in node_ids]
                    centers = np.array(centers)
                    center = np.mean(centers, axis=0)
                    # if in rotation relation - don't do it.
                    cur_mat = get_direction_with_edge_center_array(parts_graph, node_ids)
                    for node_id in node_ids:
                        cur_center = parts_graph.nodes[node_id]['part']['obb'].get_center()
                        delta = cur_center - center
                        delta = delta / np.linalg.norm(delta)
                        dot_prods = (cur_mat * delta[None, :]).sum(axis=1)
                        max_ind = np.argmax(dot_prods)
                        name = rotation_map[(max_ind, 4)]
                        parts_graph.nodes[node_id]['part']['label'] = f"{label}_{name}"
                elif len(node_ids) == 8:

                    centers = [parts_graph.nodes[node_id]['part']['obb'].get_center() for node_id in node_ids]
                    centers = np.array(centers)
                    center = np.mean(centers, axis=0)
                    # if in rotation relation - don't do it.
                    cur_mat = get_direction_with_edge_center_array(parts_graph, node_ids)
                    for node_id in node_ids:
                        cur_center = parts_graph.nodes[node_id]['part']['obb'].get_center()
                        delta = cur_center - center
                        delta = delta / np.linalg.norm(delta)
                        dot_prods = (cur_mat * delta[None, :]).sum(axis=1)
                        max_ind = np.argmax(dot_prods)
                        name = rotation_map[(max_ind, 8)]
                        parts_graph.nodes[node_id]['part']['label'] = f"{label}_{name}"
                else:
                    count = 0
                    for ind, node_id in enumerate(node_ids):
                        parts_graph.nodes[node_id]['part']['label'] = f"{label}_{count}"
                        count += 1
            elif mode == "T":
                # if "relation_ind" in cur_node['part'].keys():
                #     relation_ind = cur_node['part']['relation_ind']
                #     indices, params, sym_group_type = clean_relations[relation_ind]
                #     delta = params[0]
                #     delta = delta / np.linalg.norm(delta)
                #     dot_prods = (direction_array * delta[None, :]).sum(axis=1)
                #     max_ind = np.argmax(dot_prods)
                #     names = translation_map[(max_ind, len(node_ids))]
                #     for ind, index in enumerate(indices):
                #         parts_graph.nodes[index]['part']['label'] = f"{label}_{names[ind]}"
                # else:
                #     # check if there are ref relations
                #     ref_relations = []
                #     for relation in clean_relations:
                #         indices = relation[0]
                #         if len(indices) == 2:
                #             if all([x in node_ids for x in indices]):
                #                 ref_relations.append(relation)
                #     if len(ref_relations) > 0:
                #         # can use reflections
                #         for relation in ref_relations:
                #             indices, params, sym_group_type = relation
                #             part_1 = parts_graph.nodes[indices[0]]['part']
                #             part_2 = parts_graph.nodes[indices[1]]['part']
                #             obb_1 = part_1['obb']
                #             obb_2 = part_2['obb']
                #             center_1 = obb_1.get_center()
                #             center_2 = obb_2.get_center()
                #             center = np.mean([center_1, center_2], axis=0)
                #             delta = center_2 - center
                #             delta = delta / np.linalg.norm(delta)
                #             dot_prods = (direction_array * delta[None, :]).sum(axis=1)
                #             max_ind = np.argmax(dot_prods)
                #             direction = direction_map[max_ind]
                #             part_2['label'] = f"{label}_{direction}"
                #             part_1['label'] = f"{label}_{inverse_direction[direction]}"
                #     else:
                        # this can be wrong
                        part_1 = parts_graph.nodes[node_ids[0]]['part']
                        part_2 = parts_graph.nodes[node_ids[-1]]['part']
                        obb_1 = part_1['obb']
                        obb_2 = part_2['obb']
                        center_1 = obb_1.get_center()
                        center_2 = obb_2.get_center()
                        delta = center_2 - center_1
                        delta = delta / np.linalg.norm(delta)
                        dot_prods = (direction_array * delta[None, :]).sum(axis=1)
                        max_ind = np.argmax(dot_prods)
                        names = translation_map[(max_ind, len(node_ids))]

                        centers = [parts_graph.nodes[node_id]['part']['obb'].get_center() for node_id in node_ids]
                        centers = np.array(centers)
                        center = np.mean(centers, axis=0)
                        node_to_dot = dict()
                        for node_ind in node_ids:
                            cur_center = parts_graph.nodes[node_ind]['part']['obb'].get_center()
                            cur_delta = cur_center - center
                            # delta = delta / np.linalg.norm(delta)
                            dot_prod = np.dot(cur_delta, delta)
                            node_to_dot[node_ind] = dot_prod
                        # return sorted keys
                        indices = sorted(node_to_dot, key=lambda x: node_to_dot[x])
                        for ind, index in enumerate(indices):
                            parts_graph.nodes[index]['part']['label'] = f"{label}_{names[ind]}"
                

        stack.extend(successors)

# 
# direction_with_edge_center_array = np.array([
        
#     [0, -1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, -1],
#     [-1, 0, 0],
#     [1, 0, 0],
#     [0, -1, 1],
#     [1, -1, 0],
#     [0, -1, -1],
#     [-1, -1, 0],
#     [0, 1, 1],
#     [1, 1, 0],
#     [0, 1, -1],
#     [-1, 1, 0],
#     [-1, 0, 1],
#     [1, 0, 1],
#     [1, 0, -1],
#     [-1, 0, -1]])
        
def get_direction_with_edge_center_array(part_graph, node_ids):
    meshes = [part_graph.nodes[node_id]['part']['mesh'] for node_id in node_ids]
    gen_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        gen_mesh += mesh

    # get abb
    abb = gen_mesh.get_axis_aligned_bounding_box()
    center = abb.get_center()
    min_bound = abb.get_min_bound()
    max_bound = abb.get_max_bound()
    size = max_bound - min_bound
    size_x, size_y, size_z = size

    n_parts = len(node_ids)
    if n_parts == 8:
        direction_with_edge_center_array = np.array([
            [size_x/2, size_y/2, size_z/2], # top front right 
            [-size_x/2, size_y/2, size_z/2], # top front left
            [-size_x/2, size_y/2, -size_z/2], # top back left
            [size_x/2, size_y/2, -size_z/2], # top back right
            [size_x/2, -size_y/2, size_z/2], # bottom front right
            [-size_x/2, -size_y/2, size_z/2], # bottom front left
            [-size_x/2, -size_y/2, -size_z/2], # bottom back left
            [size_x/2, -size_y/2, -size_z/2], # bottom back right
        ])
    else:
        direction_with_edge_center_array = np.array([
            [0, -size_y/2, 0],
            [0, size_y/2, 0],
            [0, 0, size_z/2],
            [0, 0, -size_z/2],
            [-size_x/2, 0, 0],
            [size_x/2, 0, 0],
            # faces
            [0, -size_y/2, size_z/2],
            [size_x/2, -size_y/2, 0],
            [0, -size_y/2, -size_z/2],
            [-size_x/2, -size_y/2, 0],
            [0, size_y/2, size_z/2],
            [size_x/2, size_y/2, 0],
            [0, size_y/2, -size_z/2],
            [-size_x/2, size_y/2, 0],
            [-size_x/2, 0, size_z/2],
            [size_x/2, 0, size_z/2],
            [size_x/2, 0, -size_z/2],
            [-size_x/2, 0, -size_z/2]]
            )
    direction_with_edge_center_array = direction_with_edge_center_array / np.linalg.norm(direction_with_edge_center_array, axis=1, keepdims=True)
    return direction_with_edge_center_array
