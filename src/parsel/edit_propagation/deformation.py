import numpy as np
import math

class Face:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def contains_point_ids(self, i, j):
        v_id = self.vertex_ids()
        return i in v_id and j in v_id

    def other_point(self, i, j):
        v_id = set(self.vertex_ids())
        v_id.remove(i)
        v_id.remove(j)
        return v_id.pop()

    def vertex_ids(self):
        return [self.v1, self.v2, self.v3]

    def off_string(self):
        return "3 " + str(self.v1) + " " + str(self.v2) + " " + str(self.v3)
    
def inf_norm(matrix):
    return np.amax(np.abs(matrix))

def angle_between(vector_a, vector_b):
    costheta = vector_a.dot(vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))
    return math.acos(costheta)

def cot(theta):
    return math.cos(theta) / math.sin(theta)


class Deformer:

    def __init__(self, verts, vert_prime, faces):
        self.POWER = float('Inf')
        self.verts = verts
        self.verts_prime = vert_prime 
        self.verts_to_face = [[] for x in range(verts.shape[0])]
        self.faces = []
        self.n = verts.shape[0]
        number_of_faces = faces.shape[0]
        self.neighbour_matrix = np.zeros((self.n, self.n))
        self.edge_matrix = np.zeros((self.n, self.n))

        for i in range(number_of_faces):
            face_line = faces[i]
            v1_id = int(face_line[0])
            v2_id = int(face_line[1])
            v3_id = int(face_line[2])
            self.faces.append(Face(v1_id, v2_id, v3_id))
            # Add this face to each vertex face map
            self.assign_values_to_neighbour_matrix(v1_id, v2_id, v3_id)
            self.verts_to_face[v1_id].append(i)
            self.verts_to_face[v2_id].append(i)
            self.verts_to_face[v3_id].append(i)

    
        for row in range(self.n):
            self.edge_matrix[row][row] = self.neighbour_matrix[row].sum()

        self.cell_rotations = np.zeros((self.n, 3, 3))


        # print("Generating Weight Matrix")
        self.weight_matrix = np.zeros((self.n, self.n), dtype=float)
        self.weight_sum = np.zeros((self.n, self.n), dtype=float)

        for vertex_id in range(self.n):
            neighbours = self.neighbours_of(vertex_id)
            for neighbour_id in neighbours:
                self.assign_weight_for_pair(vertex_id, neighbour_id)
        # print(self.weight_matrix)
        self.precompute_p_i()
        self.calculate_cell_rotations()
        # print("Calculating Energy")
        self.energy = self.calculate_energy()
        
    
    def assign_values_to_neighbour_matrix(self, v1, v2 ,v3):
        self.neighbour_matrix[v1, v2] = 1
        self.neighbour_matrix[v2, v1] = 1
        self.neighbour_matrix[v1, v3] = 1
        self.neighbour_matrix[v3, v1] = 1
        self.neighbour_matrix[v2, v3] = 1
        self.neighbour_matrix[v3, v2] = 1


    def assign_weight_for_pair(self, i, j):
        if(self.weight_matrix[j, i] == 0):
            # If the opposite weight has not been computed, do so
            weightIJ = self.weight_for_pair(i, j)
        else:
            weightIJ = self.weight_matrix[j, i]
        self.weight_sum[i, i] += weightIJ * 0.5
        self.weight_sum[j, j] += weightIJ * 0.5
        self.weight_matrix[i, j] = weightIJ

    def weight_for_pair(self, i, j):
        local_faces = []
        # For every face associated with vert index I,
        for f_id in self.verts_to_face[i]:
            face = self.faces[f_id]
            # If the face contains both I and J, add it
            if face.contains_point_ids(i, j):
                local_faces.append(face)

        # Either a normal face or a boundry edge, otherwise bad mesh
        assert(len(local_faces) <= 2)

        vertex_i = self.verts[i]
        vertex_j = self.verts[j]

        # weight equation: 0.5 * (cot(alpha) + cot(beta))

        cot_theta_sum = 0
        for face in local_faces:
            other_vertex_id = face.other_point(i, j)
            vertex_o = self.verts[other_vertex_id]
            theta = angle_between(vertex_i - vertex_o, vertex_j - vertex_o)
            cot_theta_sum += cot(theta)
        return cot_theta_sum * 0.5
    
    def calculate_cell_rotations(self):
        # print("Calculating Cell Rotations")
        for vert_id in range(self.n):
            rotation = self.calculate_rotation_matrix_for_cell(vert_id)
            self.cell_rotations[vert_id] = rotation

    def calculate_rotation_matrix_for_cell(self, vert_id):
        covariance_matrix = self.calculate_covariance_matrix_for_cell(vert_id)

        U, s, V_transpose = np.linalg.svd(covariance_matrix)

        # U, s, V_transpose
        # V_transpose_transpose * U_transpose


        rotation = V_transpose.transpose().dot(U.transpose())
        if np.linalg.det(rotation) <= 0:
            U[:0] *= -1
            rotation = V_transpose.transpose().dot(U.transpose())
        return rotation

    def precompute_p_i(self):
        self.P_i_array = []
        for i in range(self.n):
            vert_i = self.verts[i]
            neighbour_ids = self.neighbours_of(i)
            number_of_neighbours = len(neighbour_ids)

            P_i = np.zeros((3, number_of_neighbours))

            for n_i in range(number_of_neighbours):
                n_id = neighbour_ids[n_i]

                vert_j = self.verts[n_id]
                P_i[:, n_i] = (vert_i - vert_j)
            self.P_i_array.append(P_i)

    def calculate_covariance_matrix_for_cell(self, vert_id):
        # s_i = P_i * D_i * P_i_prime_transpose
        vert_i_prime = self.verts_prime[vert_id]

        neighbour_ids = self.neighbours_of(vert_id)
        number_of_neighbours = len(neighbour_ids)

        D_i = np.zeros((number_of_neighbours, number_of_neighbours))

        P_i = self.P_i_array[vert_id]
        P_i_prime = np.zeros((3, number_of_neighbours))

        for n_i in range(number_of_neighbours):
            n_id = neighbour_ids[n_i]

            D_i[n_i, n_i] = self.weight_matrix[vert_id, n_id]

            vert_j_prime = self.verts_prime[n_id]
            P_i_prime[:, n_i] = (vert_i_prime - vert_j_prime)

        P_i_prime = P_i_prime.transpose()
        return P_i.dot(D_i).dot(P_i_prime)
    
    # Returns a set of IDs that are neighbours to this vertexID (not including the input ID)
    def neighbours_of(self, vert_id):
        neighbours = []
        for n_id in range(self.n):
            if(self.neighbour_matrix[vert_id, n_id] == 1):
                neighbours.append(n_id)
        return neighbours


    def calculate_energy(self):
        total_energy = 0
        for i in range(self.n):
            total_energy += self.energy_of_cell(i)
        return total_energy

    def energy_of_cell(self, i):
        neighbours = self.neighbours_of(i)
        total_energy = 0
        for j in neighbours:
            w_ij = self.weight_matrix[i, j]
            e_ij_prime = self.verts_prime[i] - self.verts_prime[j]
            e_ij = self.verts[i] - self.verts[j]
            r_i = self.cell_rotations[i]
            value = e_ij_prime - r_i.dot(e_ij)
            if(self.POWER == float('Inf')):
                norm_power = inf_norm(value)
            else:
                norm_power = np.power(value, self.POWER)
                norm_power = np.sum(norm_power)
            # total_energy += w_ij * np.linalg.norm(, ord=self.POWER) ** self.POWER
            total_energy += w_ij * norm_power
        return total_energy
