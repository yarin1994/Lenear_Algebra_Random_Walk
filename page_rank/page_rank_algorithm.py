from math import sqrt

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp


def create_graph_first_type(q, number_of_nodes):
    if q == 0:
        raise Exception('the q  value is 0, meaning no edges will be added to the graph')

    graph = nx.DiGraph()
    graph.add_nodes_from(range(0, number_of_nodes))
    add_edges_from_i_to_j_probabilty_q(graph, q)

    return graph


def create_graph_second_type(list_of_nodes_probabilities, number_of_nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(0, number_of_nodes))
    add_edges_from_i_to_j_probability_qj(graph, list_of_nodes_probabilities)

    return graph


def add_edges_from_i_to_j_probability_qj(graph, list_of_nodes_probabilities):
    print('adding edges...')
    for i in range(0, graph.number_of_nodes()):
        for j in range(0, graph.number_of_nodes()):
            qj = list_of_nodes_probabilities[j]
            probability = random.uniform(0, 1)

            if probability <= qj:
                graph.add_edge(i, j)


def add_edges_from_i_to_j_probabilty_q(graph, q):
    print('adding edges...')
    for i in range(0, graph.number_of_nodes()):
        for j in range(0, graph.number_of_nodes()):
            probability = random.uniform(0, 1)
            if probability <= q:
                graph.add_edge(i, j)


class RunAlgorithm:
    def __init__(self, graphs, list_of_p, list_of_epsilon, list_of_k, number_of_nodes):
        self._graphs = graphs
        self._list_of_p = list_of_p
        self._list_of_epsilon = list_of_epsilon
        self._list_of_k = list_of_k
        self.runs_data = []
        self._number_of_nodes = number_of_nodes

    def _page_rank_algorithm(self, graph, t, p, N):
        print('starting page rank algorithm with values: graph = {}, t = {}, p = {}, N = {}'.format(graph.name, t, p, N))
        d = np.zeros(self._number_of_nodes)
        start_node = random.choice([i for i in range(graph.number_of_nodes())])
        d[start_node] += 1
        current_node = start_node

        for i in range(0, t):

            if t > pow(2, 16):
                raise Exception('t is to big, canceling the run')

            for j in range(0, int(N)):
                probability = random.uniform(0, 1)
                current_node_neighbors = list(graph.neighbors(current_node))
                # print('node = {}, neighbors = {}'.format(current_node, current_node_neighbors))
                # going to a random node in the graph
                if probability <= p:
                    current_node = random.choice([i for i in range(graph.number_of_nodes())])

                # going to a random neighbor of the current node
                else:
                    if len(current_node_neighbors) != 0:
                        current_node = random.choice(current_node_neighbors)
                    else:
                        current_node = random.choice([i for i in range(graph.number_of_nodes())])

            # print('finished current walk at node {}'. format(current_node))
            d[current_node] += 1

        return d / t

    def run(self):
        for graph in self._graphs:
            for p in self._list_of_p:
                N = 1/p
                for e in self._list_of_epsilon:
                    try:
                        self._run_algorithm_first_base_case(graph, e, N, p)

                    except Exception as e:
                        print(e)

                    print(self.runs_data[-1])

                for k in self._list_of_k:
                    try:
                        self._run_algorithm_second_base_case(graph, k, N, p)

                    except Exception as e:
                        print(e)

                    print(self.runs_data[-1])

        self._write_results_to_text_file()

    def _write_results_to_text_file(self):
        with open('results.txt', 'w') as f:
            for line in self.runs_data:
                f.write(str(line))
                f.write('\n')



    def _run_algorithm_first_base_case(self, graph, e, N, p):
        print('running algorithm first base case with graph = {}, e = {}, N = {}, p = {}'.format(graph.name, e, N, p))
        t = pow(2, 1)
        previous_i_vector = np.zeros(self._number_of_nodes)
        continue_running = True
        number_of_nodes_stayed_steady_ranking = 0

        while(continue_running):
            current_i_vector = self._page_rank_algorithm(graph, t, p, N)

            if self._is_first_base_case(e, previous_i_vector, current_i_vector):
                number_of_nodes_stayed_steady_ranking = self._get_number_of_nodes_which_are_steady(previous_i_vector, current_i_vector)
                continue_running = False

            else:
                previous_i_vector = current_i_vector
                t *= 2

        self.runs_data.append([{'graph name': graph.name}, {'type of base case': 'first base case with epsilon'}, {'p': p}, {'N': N}, {'t': t}, {'epsilon': e}, {'number_of_nodes_steady': number_of_nodes_stayed_steady_ranking}])

    def _run_algorithm_second_base_case(self, graph, k, N, p):
        print('running algorithm second base case with graph = {}, k = {}, N = {}, p = {}'.format(graph.name, k, N, p))
        t = pow(2, 1)
        previous_i_vector = np.zeros(self._number_of_nodes)
        continue_running = True

        while(continue_running):
            current_i_vector = self._page_rank_algorithm(graph, t, p, N)

            if self._is_second_base_case(k, previous_i_vector, current_i_vector):
                e = np.linalg.norm(current_i_vector - previous_i_vector)
                continue_running = False

            else:
                previous_i_vector = current_i_vector
                t *= 2

        self.runs_data.append([{'graph name': graph.name}, {'type of base case': 'second base case with k'}, {'p': p}, {'N': N}, {'t': t}, {'k': k},
                               {'epsilon': e}])

    def _is_first_base_case(self, e, v1, v2):

        magnitude = np.linalg.norm(v2 - v1)

        return magnitude < e

    def _is_second_base_case(self, k, v1, v2):

        return self._get_number_of_nodes_which_are_steady(v1, v2) >= k

    def _get_number_of_nodes_which_are_steady(self, v1, v2):
        v1_sorted = np.flip(np.argsort(v1))
        v2_sorted = np.flip(np.argsort(v2))
        count = 0

        size = len(v1_sorted)
        for i in range(0, size):
            if v1_sorted[i] == v2_sorted[i]:
                count += 1

            else:
                break

        return count

    def print_runs_data(self):
        for datum in self.runs_data:
            print(datum)


def generate_graphs():
    graph_list = []
    number_of_nodes = pow(2, 9)

    # creating the first-family graph
    q_list_first_family = [1 / pow(2, 12), 1 / pow(2, 9), 1 / pow(2, 4)]
    for q in q_list_first_family:
        name = 'first_family_q=' + str(q)
        file_name = '{}_nodes={}.txt'.format(name, number_of_nodes, q)
        # graph = create_graph_first_type(q, number_of_nodes)
        # graph.name = name
        # nx.write_gpickle(graph, file_name)
        graph = nx.read_gpickle(file_name)
        graph_list.append(graph)

    # creating the second-family graph
    list_of_nodes_probabilities1 = [1/i for i in range(1, number_of_nodes)]
    list_of_nodes_probabilities1.insert(0, 1)
    list_of_nodes_probabilities2 = [1/sqrt(i) for i in range(1, number_of_nodes)]
    list_of_nodes_probabilities2.insert((0, 1))
    graph1 = create_graph_second_type(list_of_nodes_probabilities1, number_of_nodes)
    graph2 = create_graph_second_type(list_of_nodes_probabilities2, number_of_nodes)
    name_graph_1 = 'second_family_q=1/i'
    name_graph_2 = 'second_family_q=q/sqrt(i)'
    file_name1 = '{}_nodes_={}.txt'.format(name_graph_1, number_of_nodes)
    file_name2 = '{}_nodes_={}.txt'.format(name_graph_2, number_of_nodes)
    nx.write_gpickle(graph1, file_name1)
    nx.write_gpickle(graph2, file_name2)
    graph1 = nx.read_gpickle(file_name1)
    graph2 = nx.read_gpickle(file_name2)
    graph_list.append(graph1)
    graph_list.append(graph2)

    return graph_list, number_of_nodes


def main():
    graphs, number_of_nodes = generate_graphs()
    p_list = [1/2, 1/4, 1/8, 1/16, 1/32]
    epsilon_list = [1/2, 1/4, 1/8, 1/16, 1/32]
    k_list = [2, 4, 8, 16, 32]

    algorithm = RunAlgorithm(graphs, p_list, epsilon_list, k_list, number_of_nodes)
    algorithm.run()


    # probability_vector = page_rank_algorithm(graph, t, p, N)
    # print(probability_vector)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
