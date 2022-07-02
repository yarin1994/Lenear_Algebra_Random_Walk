import networkx as nx
import matplotlib.pyplot as plt
import random
import operator

p = 1/64
e = 0.2
N = 5
n = pow(2, 10)
t = 1


#select random graph using gnp_random_graph() function of networkx
G = nx.gnp_random_graph(n, p, directed=True)
degree = sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())
print(degree)
nx.draw(G, with_labels=True, node_color='green') #draw the network graph 
plt.figure(figsize=(15,10))
plt.show() #to show the graph by plotting it

# random_node is the start node selected randomly
random_node = random.choice([i for i in range(G.number_of_nodes())])
dict_counter = {} #initialise the value for all nodes as 0
for i in range(G.number_of_nodes()):
    dict_counter[i] = 0

# increment by traversing through all neighbors nodes
dict_counter[random_node] = dict_counter[random_node]+1

#Traversing through the neighbors of start node
for i in range(t):
    for j in range(N):
        list_for_nodes = list(G.neighbors(random_node))
        prob = random.uniform(0, 1)
        if e != 0 and prob <= e : # random node
            random_node = random.choice([i for i in range(G.number_of_nodes())])
            #dict_counter[random_node] = dict_counter[random_node]+1
            
        else: #neighboor
            random_node = random.choice(list_for_nodes) #choose a node randomly from neighbors
            #dict_counter[random_node] = dict_counter[random_node] + 1

    dict_counter[random_node] = dict_counter[random_node] + 1     
# using pagerank() method to provide ranks for the nodes        
rank_node = nx.pagerank(G)


#sorting the values of rank and random walk of respective nodes
sorted_rank = sorted(rank_node.items(), key=operator.itemgetter(1))
sorted_random_walk = sorted(dict_counter.items(), key=operator.itemgetter(1))
print(sorted_rank)
print(sorted_random_walk)
