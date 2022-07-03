import pickle
from math import sqrt
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import os


def create_graph_first_type(q, number_of_nodes):
    if q == 0:
        raise Exception('the q  value is 0, meaning no edges will be added to the graph')

    graph = nx.DiGraph()
    graph.add_nodes_from(range(0, number_of_nodes))
    add_edges_from_i_to_j_probability_q(graph, q)

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


def add_edges_from_i_to_j_probability_q(graph, q):
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

        return d / t, d

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
        self._write_results_to_bin_file()

    def _write_results_to_text_file(self):
        with open('results.txt', 'w') as f:
            for line in self.runs_data:
                f.write(str(line))
                f.write('\n')

    def _write_results_to_bin_file(self):
        # Step 2
        with open('results_bin/runs_data', 'wb') as binary_file:
            # Step 3
            pickle.dump(self.runs_data, binary_file)

    def load_results_from_bin_file(self):
        with open('results_bin/runs_data', 'rb') as binary_file:
            self.runs_data = pickle.load(binary_file)







    def _run_algorithm_first_base_case(self, graph, e, N, p):
        print('running algorithm first base case with graph = {}, e = {}, N = {}, p = {}'.format(graph.name, e, N, p))
        t = pow(2, 1)
        previous_i_vector = np.zeros(self._number_of_nodes)
        continue_running = True
        number_of_nodes_stayed_steady_ranking = 0

        while(continue_running):
            current_i_vector, actual_ranking_vector = self._page_rank_algorithm(graph, t, p, N)

            if self._is_first_base_case(e, previous_i_vector, current_i_vector):
                number_of_nodes_stayed_steady_ranking = self._get_number_of_nodes_which_are_steady(previous_i_vector, current_i_vector)
                continue_running = False

            else:
                previous_i_vector = current_i_vector
                t *= 2
        nodes_by_their_priority = np.flip(np.argsort(actual_ranking_vector))
        first_10_nodes_in_ranking = nodes_by_their_priority[:10]
        self.runs_data.append({'graph name': graph.name, 'type of base case': 'first base case with epsilon', 'p': p, 'N': N, 't': t, 'epsilon': e, 'k': number_of_nodes_stayed_steady_ranking, 'nodes ranking': first_10_nodes_in_ranking})

    def _run_algorithm_second_base_case(self, graph, k, N, p):
        print('running algorithm second base case with graph = {}, k = {}, N = {}, p = {}'.format(graph.name, k, N, p))
        t = pow(2, 1)
        previous_i_vector = np.zeros(self._number_of_nodes)
        continue_running = True

        while(continue_running):
            current_i_vector, actual_ranking_vector = self._page_rank_algorithm(graph, t, p, N)

            if self._is_second_base_case(k, previous_i_vector, current_i_vector):
                e = np.linalg.norm(current_i_vector - previous_i_vector)
                continue_running = False

            else:
                previous_i_vector = current_i_vector
                t *= 2
        nodes_by_their_priority = np.flip(np.argsort(actual_ranking_vector))
        first_10_nodes_in_ranking = nodes_by_their_priority[:10]
        self.runs_data.append({'graph name': graph.name, 'type of base case': 'second base case with k', 'p': p, 'N': N, 't': t, 'k': k,
                               'epsilon': e, 'nodes ranking': first_10_nodes_in_ranking})

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
    generate_graph_first_type(graph_list, number_of_nodes)
    generate_graph_second_type(graph_list, number_of_nodes)
    # save_graphs_to_txt_file(graph_list, 'graphs_txt')
    # graph_list.clear()
    # load_graphs_from_txt_file(graph_list, 'graphs_txt')

    return graph_list, number_of_nodes


def generate_graph_second_type(graph_list, number_of_nodes):
    list_of_nodes_probabilities1 = [1 / i for i in range(1, number_of_nodes)]
    list_of_nodes_probabilities1.insert(0, 1)
    list_of_nodes_probabilities2 = [1 / sqrt(i) for i in range(1, number_of_nodes)]
    list_of_nodes_probabilities2.insert(0, 1)
    graph1 = create_graph_second_type(list_of_nodes_probabilities1, number_of_nodes)
    graph2 = create_graph_second_type(list_of_nodes_probabilities2, number_of_nodes)
    name_graph_1 = 'second_family_q=1:i'
    name_graph_2 = 'second_family_q=q:sqrt(i)'
    graph1.name = name_graph_1
    graph2.name = name_graph_2
    graph_list.append(graph1)
    graph_list.append(graph2)


def generate_graph_first_type(graph_list, number_of_nodes):
    # creating the first-family graph
    q_list_first_family = [1 / pow(2, 12), 1 / pow(2, 9), 1 / pow(2, 4)]
    for q in q_list_first_family:
        name = 'first_family_q=' + str(q)
        graph = create_graph_first_type(q, number_of_nodes)
        graph.name = name
        graph_list.append(graph)


def save_graphs_to_txt_file(graph_list, folder_path):
    for graph in graph_list:
        file_name = '{}_nodes={}.txt'.format(graph.name, graph.number_of_nodes())
        file_path = os.path.join(folder_path, file_name)
        nx.write_gpickle(graph, file_path)


def load_graphs_from_txt_file(graph_list, folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            graph = nx.read_gpickle(file_path)
            graph_list.append(graph)


def plot_results(runs_data):
    organized_results = organize_results(runs_data)

    for graph_type in organized_results.keys():
        plot_graph_and_save_to_file(organized_results, graph_type)

    # print(organized_results)


def plot_graph_and_save_to_file(organized_results, graph_type):
    cases = organized_results.get(graph_type)

    for case in cases.keys():

        p_dictionary = cases.get(case)
        p_values = [p for p in p_dictionary.keys()]

        for p in p_dictionary.keys():
            plot_name = graph_type + '-' + case + '-' + 'p=' + str(p)
            run_values_list = p_dictionary.get(p)
            t_values = []
            e_values = []
            k_values = []
            nodes_ranking = []

            for single_run_values in run_values_list:
                t_values.append(single_run_values.get('t'))
                e_values.append(single_run_values.get('epsilon'))
                k_values.append(single_run_values.get('k'))
                nodes_ranking.append(single_run_values.get('nodes ranking'))

            bar_labels = ['e = ' + str(i) + ', k = ' + str(j) + 'ranking = ' + str(r) for i, j, r in zip(e_values, k_values, nodes_ranking)]
            # left_coordinates = [1, 2, 3, 4, 5]
            plt.bar(e_values, t_values, width=0.1, tick_label=bar_labels, color=['blue', 'orange'])
            plt.title(plot_name)
            plt.savefig('plots/' + plot_name + '.png')


def organize_results(runs_data):
    organized_results = {}

    for single_run_data in runs_data:
        cases_dictionary = organized_results.setdefault(single_run_data['graph name'], {})
        p_dictionary = cases_dictionary.setdefault(single_run_data['type of base case'], {})
        e_k_t_values_list = p_dictionary.setdefault(single_run_data['p'], [])
        e_k_t_values_list.append({'epsilon': single_run_data['epsilon'], 'k': single_run_data['k'], 't': single_run_data['t'], 'nodes ranking': single_run_data['nodes ranking']})
        # e_k_t_values_dictionary.update([('epsilon', single_run_data['epsilon']), ('k', single_run_data['k']), ('t', single_run_data['t'])])

    return organized_results


def main():
    graphs, number_of_nodes = generate_graphs()
    p_list = [1/2, 1/4, 1/8, 1/16, 1/32]
    epsilon_list = [1/2, 1/4, 1/8, 1/16, 1/32]
    k_list = [2, 4, 8, 16, 32]
    algorithm = RunAlgorithm(graphs, p_list, epsilon_list, k_list, number_of_nodes)
    # number_of_nodes = 32
    # graphs = []
    # generate_graph_first_type(graphs, 32)
    # p_test_list = [1/2, 1/4]
    # epsilon_test_list = [1/4, 1/8]
    # k_test_list = [2]
    # algorithm = RunAlgorithm(graphs, p_list, epsilon_list, k_list, number_of_nodes)

    algorithm.run()
    algorithm.runs_data.clear()
    algorithm.load_results_from_bin_file()
    print(algorithm.runs_data)
    plot_results(algorithm.runs_data)


    # probability_vector = page_rank_algorithm(graph, t, p, N)
    # print(probability_vector)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
