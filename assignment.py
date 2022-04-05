# Lochlann Hackett - 261056757
# Cian Ó Gaora - 261056892

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parse data from text file
with open('input.txt') as f:
    lines = f.readlines()

data = []

for line in lines:
    if line[0] != '#':
        data.append(line)

formatted_data = []
count = 0

for i in data:
    if count == 1:
        nums = [float(n) for n in i.split()]
        formatted_data.append(nums)
    else:
        nums = [int(n) for n in i.split()]
        formatted_data.append(nums)
    count += 1

f.close()

# create global variables to hold number of cities, reliabilies, and costs
n_cities = formatted_data[0][0]
reliabilities = formatted_data[1]
costs = formatted_data[2]

relcount = 0
start = 1

rel_mat = np.zeros((6, 6))
cost_mat = np.zeros((6, 6))
relMap = []
costMap = []

# Create np arrays to represent reliability matrix and cost matrix for simple referencing
for i in range(n_cities):
    for j in range(start, n_cities):
        rel_mat[i][j] = reliabilities[relcount]
        relMap.append((i, j, reliabilities[relcount]))
        relcount += 1
    start += 1

relcount = 0
start = 1

for i in range(n_cities):
    for j in range(start, n_cities):
        cost_mat[i][j] = costs[relcount]
        costMap.append((i, j, costs[relcount]))
        relcount += 1
    start += 1


# Recursive function decomposes a graph into sub graphs
def network_reliability(G, graph, r):
    cycles = nx.cycle_basis(G, root=None)
    _, nonCycleRels = get_edges_not_in_cycle(cycles, graph, r, n_cities)
    numNonCycleEdges = len(nonCycleRels)
    cycles = nx.cycle_basis(G, root=None)
    numCycles = len(cycles)

    # R1 will be the reliability of the subgraph where the removed edge does not work
    # R2 will be the reliability of the subgraph where the removed edge does work
    # R will be the weighted average of each Rn

    R1 = 0
    R2 = 0
    R = 1

    s = 0
    count = 0
    for i in range(n_cities):
        for j in range(s, n_cities):
            if graph[i][j] >= 1.0:
                # Copy state of existing data to avoid changing data which should not be changed
                T = G.copy()
                T.remove_edge(i, j)
                t_graph = graph.copy()
                t_r = r.copy()
                t_graph[i][j] = 0
                temp_cycles = nx.cycle_basis(T, root=None)
                temp_numCycles = len(temp_cycles)
                _, temp_nonCycleRels = get_edges_not_in_cycle(temp_cycles, t_graph, r, n_cities)
                temp_numNonCycleEdges = len(temp_nonCycleRels)

                # This condition checks that after an edge is removed, num cycles is reduced, but no more noncycle edges are added.
                # As a workaround for removing nodes and updating the edges, the new reliability matrix is set to 1 at the removed
                # edge index. This has the same effect as removing a node when calculating the reliability of a cycle.

                if temp_numCycles < numCycles and temp_numNonCycleEdges <= numNonCycleEdges and r[i][j] != 1:
                    t_r[i][j] = 1
                    count += 1

                    # First recursive call. Passes in new graph with removed edge.
                    R1 = (1 - r[i][j]) * network_reliability(T, t_graph, r)

                    # Second recursive call. Passes in original graph from this call, with the reliability at this index set to 1.
                    R2 = r[i][j] * network_reliability(G, graph, t_r)

                    R = R1 + R2
                    break
                else:
                    continue
            else:  # python trick to break from nested loops
                continue
            break
        else:
            s += 1
            continue
        break

    if count == 0:  # Count will =0 if the condition never held in the for loop AKA this graph could not be decomposed
        # Thus graph cannot be further decomposed, and cycles reliabilities should by multiplied together,
        # along with non cycle reliabilities

        cycleRels = get_cycles_reliability(cycles, r)
        _, noncycleRels = get_edges_not_in_cycle(cycles, graph, r, n_cities)

        for cycle in range(len(cycleRels)):
            R *= cycleRels[cycle]
        for edge in range(len(noncycleRels)):
            R *= noncycleRels[edge]

    return R


# Return list of reliabilities of each edge in the cycle passed in
def get_cycle_edges(cycle, r):
    rels = np.zeros(len(cycle))
    for e in range(len(cycle)):
        if e == len(cycle) - 1:
            if cycle[e] < cycle[0]:
                rels[e] = r[cycle[e]][cycle[0]]
            else:
                rels[e] = r[cycle[0]][cycle[e]]
        else:
            if cycle[e] < cycle[e + 1]:
                rels[e] = r[cycle[e]][cycle[e + 1]]
            else:
                rels[e] = r[cycle[e + 1]][cycle[e]]

    return rels


def get_edges_not_in_cycle(cycles, graph, r, numcities):
    noncycleEdges = np.zeros((6, 6))
    rels = []

    # If there are no cycles then add all edges of graph to list
    if len(cycles) == 0:
        noncycleEdges = graph.copy()

        for i in range(len(graph[0])):
            for j in range(len(graph[0])):
                if noncycleEdges[i][j] >= 1:
                    rels.append(r[i][j])

    # for each cycle in the list of cycles, determine which edges of graph aren't in the cycle
    else:
        for k in range(len(cycles)):
            for i in range(numcities):
                for j in range(numcities):
                    if i in cycles[k] and j in cycles[k] or graph[i][j] == 0:
                        noncycleEdges[i][j] = 1

        # Determine reliabilities of non-cycle edges
        for i in range(len(graph[0])):
            for j in range(len(graph[0])):
                if noncycleEdges[i][j] == 0:
                    rels.append(r[i][j])

    return noncycleEdges, rels


# Calculate reliability of a cycle
def get_cycles_reliability(cycles, r):
    l = len(cycles)
    # print(cycles)
    rel = np.zeros(l)
    for i in range(l):
        edges = get_cycle_edges(cycles[i], r)
        # print(edges)
        # Add reliability for when one edge in cycle does not work
        for edge in range(len(edges)):
            r_it = 1
            for edge2 in range(len(edges)):
                if edge == edge2:
                    r_it *= (1 - edges[edge])
                else:
                    r_it *= edges[edge2]
            rel[i] += r_it
        # Add reliability for when all edges work
        r_it = 1
        for edge in range(len(edges)):
            r_it *= edges[edge]

        rel[i] += r_it
    return rel


# Find the edge which returns the best rel:cost ratio, and return edge and new reliability of network
def find_best(G, graph, r, c):
    cycles = nx.cycle_basis(G, root=None)
    rel = get_cycles_reliability(cycles, r)

    noncycleEdges, noncycleRels = get_edges_not_in_cycle(cycles, graph, r, n_cities)
    H = G.copy()
    t_graph = graph.copy()
    cycles = nx.cycle_basis(H, root=None)
    l = len(cycles)
    new_r = 0
    ratio = 0
    best_ratio, besti, bestj, rest_rel = 0, 0, 0, 0

    start = 1
    r_old = network_reliability(H, t_graph, r)
    print(r_old)

    # For each edge in reliability matrix, add edge and calculate reliability to cost ratio
    added_parallel = False
    best_edge_parallel = False
    r_old = network_reliability(H, t_graph, r)
    # print(r_old)
    #
    for i in range(n_cities):
        for j in range(start, n_cities):
            H = G.copy()
            t_graph = graph.copy()
            r_ = r.copy()
            if t_graph[i][j] >= 1.0:  # If adding parallel edge to already present edge
                r_[i][j] = (r_[i][j] + rel_mat[i][j]) - (r_[i][j]*rel_mat[i][j])
                added_parallel = True
            else:
                H.add_edge(i, j)
                t_graph[i][j] += 1
                added_parallel = False

            # Calculate new reliability and ratio
            new_r = network_reliability(H, t_graph, r_)
            ratio = (new_r - r_old) / c[i][j]

            # Assign best edge based on improvement in ratio
            if ratio > best_ratio:
                best_ratio = ratio
                besti, bestj, best_rel = i, j, new_r
                best_edge_parallel = added_parallel
        start += 1

    return besti, bestj, best_rel, best_edge_parallel


# Generate mst, calculate reliability and cost
def connect(visited):
    # Adjacency matrix to represent graph
    graph = np.zeros((6, 6))

    # Sort reliabilities by highest value
    ind = np.dstack(np.unravel_index(np.argsort((-rel_mat).ravel()), (6, 6)))
    ind = ind[0]

    edge_count = 0

    # Add edges to adjacency matrix, visited checks if edge has already been attempted
    for edge in range(18):
        if visited[ind[edge][0]] and visited[ind[edge][1]]:
            continue
        elif edge_count == 5:
            break
        else:
            graph[ind[edge][0]][ind[edge][1]] = 1

            edge_count += 1
            visited[ind[edge][0]] = True
            visited[ind[edge][1]] = True

    # print(graph)

    total_reliability = 1
    total_cost = 0

    # Calculate reliability and cost
    for i in range(n_cities):
        for j in range(n_cities):
            if graph[i][j] == 1.0:
                total_reliability *= rel_mat[i][j]
                total_cost += cost_mat[i][j]

    return graph, total_reliability, total_cost


# Convenient function to get list of edges based on adjacency matrix
def get_edges_from_adj_matrix(am):
    edges = []
    for i in range(n_cities):
        for j in range(n_cities):
            if am[i, j] == 1:
                edges.append([i, j])
    return edges


# Calculate cost of graph
def calc_cost(am):
    cost = 0
    for i in range(n_cities):
        for j in range(n_cities):
            if am[i, j] > 0:
                cost += am[i, j] * cost_mat[i, j]
    return cost


# Driver code to call all methods and build graph
def driver(r_target, cost_target):
    visited = [False for i in range(n_cities)]
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}

    # create mst and calculate cost and reliability
    adj_matrix, mst_reliability, mst_cost = connect(visited)

    if mst_cost > cost_target:
        print(f"Cities cannot be connected for cost {cost_target}")
        return
    if mst_reliability >= r_target:
        # Have a function to draw graph of network here
        print("cities can be connected with target reliablility and cost")

    # create nx graph to take advantage of built-in functionality
    graph_nx = nx.Graph()

    total_R = 1
    n_R = 0
    total_C = calc_cost(adj_matrix)
    edges = get_edges_from_adj_matrix(adj_matrix)
    graph_nx.add_edges_from(edges)
    rel_mat_copy = rel_mat.copy()
    it = 0

    while total_C <= cost_target and n_R <= r_target:
        best_i, best_j, n_R, flag = find_best(graph_nx, adj_matrix, rel_mat_copy, cost_mat)
        print("Iteration: "+ str(it))
        print(best_i, best_j)
        adj_matrix[best_i, best_j] += 1.0

        if adj_matrix[best_i, best_j] == 1:
            graph_nx.add_edge(best_i, best_j)

        if(flag):
            rel_mat_copy[best_i][best_j] = (rel_mat_copy[best_i][best_j] + rel_mat[best_i][best_j]) - (rel_mat_copy[best_i][best_j] * rel_mat[best_i][best_j])

        total_C = calc_cost(adj_matrix)
        it+=1
        print("Reliability: " + str(n_R))

        nx.draw(graph_nx, node_size=1000, labels=labels)
        plt.show()

    if (total_C > cost_target or n_R < r_target):
        print("A network can not be made that satisfies the reliability target and cost constraint")
        print("Total cost: " + str(total_C))
        print("Network reliability: " + str(n_R))
    else:
        print("Total cost: " + str(total_C))
        print("Network reliability: " + str(n_R))

    print(n_R)

    print(adj_matrix)

# Call driver with parameters (target_reliability, cost_constraint)
# Graph will be displayed, but will not show when two nodes have multiple edges in parallel
# Refer to adjacency matrix printed at the end of the function to see the edge count between any 2 nodes
driver(0.999, 200)

