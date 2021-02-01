import numpy as np

from models.graph import tools

num_node = 10
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0,1),(2,3),(0,4),(2,4),(4,5),(4,7),(4,9),(6,7),(8,9)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph_P:
    def __init__(self):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary_with_I)
