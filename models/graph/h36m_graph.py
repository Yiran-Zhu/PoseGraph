import numpy as np

from models.graph import tools

num_node = 16
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), (9, 8), 
                                    (10, 8), (11, 10), (12, 11), (13, 8), (14, 13), (15, 14)]                                    
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary_with_I)
