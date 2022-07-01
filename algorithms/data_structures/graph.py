import numpy as np
import logging

INITIAL_VALUE_EDGE_MATRIX = 1

class Edge:
    def __init__(self, out_vertex, in_vertex):
        self.edge = []
        self.edge.append(out_vertex)
        self.edge.append(in_vertex)

    def get_out_vertex(self):

        return self.edge[0]

    def get_in_vertex(self):

        return self.edge[1]


class Graph:
    def __init__(self, number_of_vertex, edges, logger=None):
        self._neighbor_list = NeighborList(number_of_vertex, edges)
        self._neighbor_matrix = NeighborMatrix(number_of_vertex, edges)
        self._logger = None
        self._number_of_vertexes = self._neighbor_list.__get_number_of_vertexes__()

        if logger is not None:
            self.set_logger(logger)

    def __set_logger__(self, logger):
        self._logger = logger

    def __get_number_of_vertexes__(self):

        return self._number_of_vertexes
#decide how to work with logger


class NeighborList:
    def __init__(self, number_of_vertexes, edges):
        self._vertex_array, self._neighbors = self._create_vertex_array_and_neighbors_list(number_of_vertexes)
        self.__add_edges__(edges)


    # create an array of size number of vertexes, also creates a numpy matrix in sizes n*n and returns both of them
    def _create_vertex_array_and_neighbors_list(self, number_of_vertexes):
        try:
            vertex_array = np.arange(number_of_vertexes)
            neighbors_list = [[] for i in range(number_of_vertexes)]

            return vertex_array, neighbors_list

        except Exception as e:
            print(e)

        return vertex_array, neighbors_list

    def __get_number_of_vertexes__(self):

        return self._vertex_array.shape[0]


    #receives a list of coupuls of vertexes
    def __add_edges__(self, edges):
        for edge in edges:
            out_vertex = edge.get_out_vertex()
            in_vertex = edge.get_in_vertex()

            self._neighbors[out_vertex].append(in_vertex)


class NeighborMatrix:
    def __init__(self, number_of_vertexes, edges):
        self.matrix = np.zeros((number_of_vertexes, number_of_vertexes))
        self.__add_edges__(edges)

    def __add_edges__(self, edges):
        for edge in edges:
            out_vertex = edge.get_out_vertex()
            in_vertex = edge.get_in_vertex()

            self.matrix[out_vertex][in_vertex] = INITIAL_VALUE_EDGE_MATRIX



