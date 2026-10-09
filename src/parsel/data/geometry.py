import numpy as np
import copy
import open3d as o3d
import open3d.visualization as vis
from .constants import PC_SIZE, ROT_HAUS_DIST_THRESHOLD, NOISE_AMT, COSINE_THRESHOLD, CROSS_THRESHOLD, AXIS_COSINE_THRESHOLD

MIN_DIFF_FOR_INVERSION = 1e-2

BASE_CORNERS = np.array([[-1, -1, -1],
                    [1, -1,-1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1]])


def obb_to_mesh(obb, half=False, return_mat=False):

    corner_points = obb_to_corner_points(obb)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corner_points)
    # faces = np.array([
    #     [2, 1, 0],
    #     [1, 2, 7], 
    #     [0, 1, 3], 
    #     [3, 1, 6],
    #     [2, 5, 7],
    #     [5, 4, 7],
    #     [5, 3, 6],
    #     [5, 6, 4],
    #     [0, 3, 2],
    #     [3, 5, 2],
    #     [1, 7, 6],
    #     [6, 7, 4],
    #     ])
    faces = np.array([
        [0, 5, 4],
        [0, 1, 5], 
        [3, 7, 6], 
        [3, 6, 2],
        [3, 0, 4],
        [3, 4, 7],

        [2, 5, 1],
        [2, 6, 5],
        [3, 1, 0],
        [3, 2, 1],
        [4, 6, 7],
        [4, 5, 6],
        ])
    if half:
        faces = faces[::2]
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if return_mat:
        mat_box = vis.rendering.MaterialRecord()
        mat_box.shader = 'defaultLitTransparency'
        # mat_box.shader = 'defaultLitSSR'
        # mat_box.base_color = [0.95, 0.0, 0.0, 0.5]
        # mat_box.base_roughness = 0.5
        # mat_box.base_reflectance = 0.2
        # mat_box.base_clearcoat = 1.0
        # mat_box.thickness = 0.5
        # mat_box.transmission = 0.5
        # mat_box.absorption_distance = 0
        # mat_box.absorption_color = [0.25, 0.25, 0.25]
        return mesh, mat_box
    else:
        return mesh

def obb_to_corner_points(obb):
    center, size, axis = obb.center, obb.extent, obb.R
    # center = np.asarray(center)
    half_size = size / 2
    scaled_corners = BASE_CORNERS * half_size[None, :]
    vertices = scaled_corners @ axis.T
    corner_points = vertices + center[None, :]



    # corner_points = obb.get_box_points()
    corner_points = np.asarray(corner_points)
    return corner_points
    
def get_rot_trans_candidates(obb_1, obb_2):
    from hausdorff import hausdorff_distance

    t_1 = np.eye(4)
    t_1[:3, 3] = -obb_1.get_center()
    t_2 = np.eye(4)
    t_2[:3, :3] = obb_1.R.T
    t_3 = np.eye(4)

    t_3[:3, :3] = obb_2.R
    t_4 = np.eye(4)
    t_4[:3, 3] = obb_2.get_center()
    # T_1 = t_4 @ t_3 @ t_2 @ t_1
    # now for all the others
    all_transforms = []
    box_mesh_1 = obb_to_mesh(obb_1)
    box_mesh_2 = obb_to_mesh(obb_2)
    pcd_1 = box_mesh_1.sample_points_poisson_disk(number_of_points=PC_SIZE//2)
    pcd_2 = box_mesh_2.sample_points_poisson_disk(number_of_points=PC_SIZE//2)


    for axis_1 in range(3):
        for value_1 in [-1, 1]:
            axis_2_op = [0, 1, 2]
            axis_2_op.remove(axis_1)
            for axis_2 in axis_2_op:
                for value_2 in [-1, 1]:
                    vec_1 = np.zeros(3)
                    vec_1[axis_1] = value_1
                    vec_2 = np.zeros(3)
                    vec_2[axis_2] = value_2
                    vec_3 = np.cross(vec_1, vec_2)
                    vec_3 = vec_3 / np.linalg.norm(vec_3)
                    rot_mat = np.eye(4)
                    rot_mat[:3, :3] = np.array([vec_1, vec_2, vec_3]).T
                    t_3_rot = rot_mat @ t_3.copy()
                    transform = t_4 @ t_3_rot @ t_2 @ t_1
                    # now check if the transformed pcd_1 and pcd_2 are close enough
                    transformed_pcd_1 = copy.deepcopy(pcd_1).transform(transform)
                    h_dist = hausdorff_distance(np.asarray(transformed_pcd_1.points),
                                                np.asarray(pcd_2.points))
                    if h_dist < ROT_HAUS_DIST_THRESHOLD:
                        try:
                            print(f"--> Sym candidate for {axis_1} {value_1} and {axis_2} {value_2}")
                            all_transforms.append(transform)
                        except:
                            print(f"failed to get center for {axis_1} {axis_2}")
    return all_transforms

# Forward map
def get_reflection_matrix(center_point, normal_dir):
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = np.eye(3) - 2 * np.outer(normal_dir, normal_dir)
                # Convert to transform matrix
    t_1 = np.eye(4)
    t_1[:3, 3] = -center_point
    t_2 = np.eye(4)
    t_2[:4, :4] = transform_mat
    t_3 = np.eye(4)
    t_3[:3, 3] = center_point
    transform_mat = t_3 @ t_2 @ t_1
    return transform_mat

# Backward map
def invert_matrix_to_components(transform):
    rotation = np.copy(transform[:3, :3])
    # 1. Identify if its rotation or reflection
    det = np.linalg.det(rotation)

    if np.isclose(det, 1):
        # It's a rotation
        axis = rotation_to_axis_angle(rotation)
        angle = (np.linalg.norm(axis) + np.pi) % (2*np.pi) - np.pi
        if np.abs(angle) < 1e-1:
            translation = np.copy(transform[:3, 3])
            return "T", translation
        else:
            normed_axis = axis / (np.linalg.norm(axis) + 1e-16)
            origin, diff = get_origin(np.copy(transform), normed_axis, c_type="R")
            if np.linalg.norm(diff) < MIN_DIFF_FOR_INVERSION:
                return "R", axis, origin
            else:
                translation = diff
                return "R", axis, origin, translation 
    elif np.isclose(det, -1):
        # It's a reflection
        normal = reflection_plane_normal(rotation)
        normal = normal / (np.linalg.norm(normal) + 1e-16)
        origin, diff = get_origin(np.copy(transform), normal, c_type="REF")
        if np.linalg.norm(diff) < MIN_DIFF_FOR_INVERSION:
            return "REF", normal, origin
        else:
            # Lets see if this is okay...
            
            translation = diff
            return "REF", normal, origin

        
def rotation_to_axis_angle(matrix):
    import pytorch3d.transforms
    import torch as th

    # use pytorch3d implementation
    matrix_np = th.from_numpy(matrix)
    axis = pytorch3d.transforms.matrix_to_axis_angle(matrix_np)
    axis = axis.detach().cpu().numpy()
    axis = np.real(axis)
    return axis
    # self implementation
    # angle = np.arccos((np.trace(matrix) - 1) / 2)
    # x = matrix[2, 1] - matrix[1, 2]
    # y = matrix[0, 2] - matrix[2, 0]
    # z = matrix[1, 0] - matrix[0, 1]
    # axis = np.array([x, y, z])
    # axis = axis / (np.linalg.norm(axis) + 1e-16)
    # axis = axis * angle
    # return axis

def reflection_plane_normal(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    normal = eigenvectors[:, np.argmin(np.abs(eigenvalues + 1))]
    normal = np.real(normal)
    return normal

def get_origin(transform, axis, c_type="R"):
    # Attempt to solve for a point that, when rotated, does not involve translation.
    A = transform[:3, :3] - np.eye(3)
    b = -transform[:3, 3]
    A = A.astype(np.double)
    b = b.astype(np.double)
    
    epsilon = 1e-16
    A += epsilon * np.eye(3)

    origin = np.linalg.lstsq(A, b, rcond=None)[0]
    diff = np.matmul(A, origin) - b
    # then solve smaller problems and see?
    all_origins = [origin]
    all_diff_norms = [np.linalg.norm(diff)]
        # print('normal diff', diff, "solution", origin)
        # Try solving for a 2x2 matrix
    Aa = A[:2, :2]
    bb = b[:2]
    origin = np.linalg.lstsq(Aa, bb, rcond=None)[0]
    origin = np.concatenate([origin, [0]])
    diff = np.matmul(A, origin) - b
    if np.isreal(origin).all():
        all_origins.append(np.real(origin))
        all_diff_norms.append(np.linalg.norm(diff))

    # print('normal diff_1', diff, "solution", origin)
    Aa = A[1:, 1:]
    bb = b[1:]
    origin = np.linalg.lstsq(Aa, bb, rcond=None)[0]
    origin = np.concatenate([[0], origin])
    diff = np.matmul(A, origin) - b
    if np.isreal(origin).all():
        all_origins.append(np.real(origin))
        all_diff_norms.append(np.linalg.norm(diff))

    # print('normal diff_2', diff, "solution", origin)
    Aa = A[::2, ::2]
    bb = b[::2]
    origin = np.linalg.lstsq(Aa, bb, rcond=None)[0]
    origin = np.stack([origin[0], 0, origin[1]])
    diff = np.matmul(A, origin) - b
    if np.isreal(origin).all():
        all_origins.append(np.real(origin))
        all_diff_norms.append(np.linalg.norm(diff))
    # print('normal diff_3', diff, "solution", origin)
    origin = np.array([0, 0, 0])
    diff = np.matmul(A, origin) - b
    if np.isreal(origin).all():
        all_origins.append(np.real(origin))
        all_diff_norms.append(np.linalg.norm(diff))
                        # raise ValueError("Cannot find origin")
    origin = all_origins[np.argmin(all_diff_norms)]
    # Now we should get the point closer to origin
    if c_type == "R":
        actual_origin = origin - np.dot(origin, axis) * axis
    else:
        actual_origin = np.dot(origin, axis) * axis

    
    origin = actual_origin.astype(np.float32)
    diff = np.matmul(A, origin) - b
    print("difference after norm", np.linalg.norm(diff), "solution", origin)
    # Correct for diff
    return origin, diff



def get_oriented_bbox_with_compas(pcd, scale_factor=1.0):
    import compas
    from compas.geometry import Rotation as cmp_Rotation
    from compas.geometry import transform_points_numpy as cmp_transform_points_numpy

    R = cmp_Rotation.from_axis_and_angle([1.0, 0.0, 0.0], 0)
    points = cmp_transform_points_numpy(np.asarray(pcd.points), R)
    bbox_points = compas.geometry.oriented_bounding_box_numpy(points)
    bbox_center = np.mean(bbox_points, axis=0)
    bbox_points = (bbox_points - bbox_center) * scale_factor + bbox_center
    # axis_1 = bbox_points[1] - bbox_points[0]
    # axis_2 = bbox_points[3] - bbox_points[0]
    # axis_3 = bbox_points[4] - bbox_points[0]
    # extent_1 = np.linalg.norm(axis_1)
    # extent_2 = np.linalg.norm(axis_2)
    # extent_3 = np.linalg.norm(axis_3)
    # extents = [extent_1, extent_2, extent_3]
    # axis_1 = axis_1 / extent_1
    # axis_2 = axis_2 / extent_2
    # axis_3 = axis_3 / extent_3
    # rotation_matrix = [axis_1, axis_2, axis_3]
    # obb_2 = o3d.geometry.OrientedBoundingBox(center=bbox_center, R=rotation_matrix, extent=extents)
    obb = o3d.geometry.PointCloud()
    obb.points = o3d.utility.Vector3dVector(bbox_points)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(obb.points) # what if this fucks up?

    return obb

def get_oriented_bounding_box_with_fixing(pcd, scale_factor=1.0, try_aabb=True):
    try: 
        obb = get_oriented_bbox_with_compas(pcd, scale_factor=scale_factor)
    except:
        # In this case add normal noise.
        # TODO: More sophisticated noise - maybe add a plane and then add noise
        points = np.asarray(pcd.points)
        if len(pcd.points) < 4:
            # doesn't handle 0 or 1 points.
            n_points = 4 - len(pcd.points)
            points_2 = np.random.uniform(low=-NOISE_AMT, high=NOISE_AMT, size=(n_points, 3))
            points_2 += np.mean(points, axis=0)
            points = np.concatenate([points, points_2], axis=0)
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            points = points + np.random.uniform(low=-NOISE_AMT, high=NOISE_AMT, size=points.shape)
            pcd.points = o3d.utility.Vector3dVector(points)
        obb = get_oriented_bbox_with_compas(pcd, scale_factor=scale_factor)
    # get axis aligned bb
    if try_aabb:
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        # onvert to obb
        aabb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
        # measure the ratio
        aabb_vol = aabb.volume()
        obb_vol = obb.volume()
        ratio_1 = obb_vol / aabb_vol
        ratio_2 = aabb_vol / obb_vol
        if min(ratio_1, ratio_2) > 0.90:
            return aabb
        else:
            return obb
    else:
        return obb

def axis_close(vec_a, vec_b):
    vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-16)
    vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-16)
    cosine = np.dot(vec_a, vec_b)
    if np.abs(cosine) > AXIS_COSINE_THRESHOLD:
        return True
    else:
        return False
    
def axis_close(axis_a, axis_b, origin_a, origin_b):
    axis_a = axis_a / (np.linalg.norm(axis_a) + 1e-16)
    axis_b = axis_b / (np.linalg.norm(axis_b) + 1e-16)
    cosine = np.dot(axis_a, axis_b)
    if cosine > AXIS_COSINE_THRESHOLD:
        # Version 2 - comaring points
        vec_a = origin_a - origin_b
        diff = np.linalg.norm(vec_a)
        if diff < CROSS_THRESHOLD:
            return True
        else:
            return False
    
def line_close(axis_a, axis_b, origin_a, origin_b):
    axis_a = axis_a / (np.linalg.norm(axis_a) + 1e-16)
    axis_b = axis_b / (np.linalg.norm(axis_b) + 1e-16)
    cosine = np.dot(axis_a, axis_b)
    if np.abs(cosine) > COSINE_THRESHOLD:
        # Now check if the origin_a is in line with axis_b and origin b
        # (origin_b - origin_a) cross with axis_b should be close to 0 
        # vec_a = origin_a - origin_b
        # vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-16)
        # cross = np.cross(vec_a, axis_b)
        # amount = np.linalg.norm(cross)
        # if  amount < CROSS_THRESHOLD:
        #     return True
        # Version 2 - comaring points
        vec_a = origin_a - origin_b
        diff = np.linalg.norm(vec_a)
        if diff < CROSS_THRESHOLD:
            return True
        else:
            return False
        
def get_origin_2(transform):
    A = transform[:3, :3] - np.eye(3)
    b = -transform[:3, 3]
    
    # Check the condition number of A
    cond_number = np.linalg.cond(A)

    # Set a threshold for the condition number
    # If the condition number is too high, the matrix is nearly singular and the results may not be reliable.
    cond_threshold = 1e12  # This threshold can be adjusted based on empirical observations

    if cond_number < cond_threshold:
        # If the condition number is acceptable, solve the least squares problem
        origin, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        origin = origin.astype(np.float32)
    else:
        # If the matrix is nearly singular, handle accordingly
        # For example, return a default value or indicate an error
        origin = np.array([0, 0, 0], dtype=np.float32)  # Default value or handle as appropriate

    return origin

def get_origin_3(transform, zero_axis=None):
    A = transform[:3, :3] - np.eye(3)
    b = -transform[:3, 3]

    if zero_axis is not None:
        # Set the specified axis to zero
        A[zero_axis, :] = 0
        A[:, zero_axis] = 0
        A[zero_axis, zero_axis] = 1
        b[zero_axis] = 0

    origin, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    origin = origin.astype(np.float32)
    return origin

def get_origin_4(transform):
    
    # solve 1
    sol_1 = get_origin_3(transform, zero_axis=None)
    sol_2 = get_origin_3(transform, zero_axis=0)
    sol_3 = get_origin_3(transform, zero_axis=1)
    sol_4 = get_origin_3(transform, zero_axis=2)
    sols = [sol_1, sol_2, sol_3, sol_4]
    dists = [np.linalg.norm(sol) for sol in sols]
    min_dist = np.min(dists)
    min_idx = np.argmin(dists)
    return sols[min_idx]


def get_origin_5(transform, axis):
    
    axis = axis / np.linalg.norm(axis)

    # Extract rotation matrix and translation vector
    b = transform[:3, 3]

    # Normalize the axis vector

    # Solve for the point on the axis (p = t + lambda * axis)
    # where t is any point on the translation vector
    # and lambda is a scalar that we solve for
    lambda_val = np.dot(b, axis) / (np.dot(axis, axis) + 1e-16)
    origin = b - lambda_val * axis

    return origin.astype(np.float32)

def extract_normals_and_sizes(obb_1):
    # get the 3 face normals
    points = obb_1.get_box_points()
    # 2, 1, 0
    normal_1 = np.cross(points[2] - points[1], points[1] - points[0])
    # 0, 1, 3
    normal_2 = np.cross(points[0] - points[1], points[1] - points[3])
    # 0, 3, 2
    normal_3 = np.cross(points[0] - points[3], points[3] - points[2])
    normal_1 = normal_1 / np.linalg.norm(normal_1)
    normal_2 = normal_2 / np.linalg.norm(normal_2)
    normal_3 = normal_3 / np.linalg.norm(normal_3)
    # now create reflection about the center
    normals = [normal_1, normal_2, normal_3]
    size_1 = np.linalg.norm(points[0] - points[3])
    size_2 = np.linalg.norm(points[0] - points[2])
    size_3 = np.linalg.norm(points[0] - points[1])
    sizes = [size_1, size_2, size_3]
    return normals, sizes
