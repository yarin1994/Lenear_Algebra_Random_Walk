from data_structures.graph import *
import numpy as np
import enum

class WalkingOption(enum.Enum):
    random_neighbor = 0
    random_vertex = 1

class PageRank:
    def __init__(self, graph, N, t, p):
        self._graph = graph
        self._number_of_vertexes = graph.__get_number_of_vertexes__()
        self._N = N
        self._t = t
        self._p = p
        self._d = np.zeros(1, self._number_of_vertexes)

    def __run_random_walk__(self, vertex_to_start_from):
        current_vertex = vertex_to_start_from

        for j in range(0, self._t):
            for k in range(0, self._N):
                walking_option = self._get_walking_option()

                if walking_option == WalkingOption.random_vertex:
                    current_vertex = self._walk_to_random_vertex()

                else: #walking option == random_neighbor
                    current_vertex = self._walk_to_random_neighbor()

            self._d[current_vertex] += 1


    def _get_walking_option(self):
        pass

    def _walk_to_random_vertex(self):
        pass

    def _walk_to_random_neighbor(self):
        pass

def create_graph_first_family():
    pass

def create_graph_second_family():
    pass








