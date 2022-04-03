import numpy as np
import networkx as nx

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

n_cities = formatted_data[0][0]
reliabilities = formatted_data[1]
costs = formatted_data[2]

relcount = 0
start = 1

rel_mat = np.zeros((6, 6))
cost_mat = np.zeros((6, 6))
relMap = []
costMap = []

for i in range(n_cities):
    for j in range(start, n_cities):
        rel_mat[i][j] = reliabilities[relcount]
        relMap.append((i, j, reliabilities[relcount]))
        # print(rel_mat[i][j])
        relcount += 1
    start += 1

relcount = 0
start = 1

for i in range(n_cities):
    for j in range(start, n_cities):
        cost_mat[i][j] = costs[relcount]
        # print(cost_mat[i][j])
        costMap.append((i, j, costs[relcount]))
        relcount += 1
    start += 1


# print(costMap)

def add_v(v, graph):
    if v in graph:
        return
    else:
        graph[v] = []


def add_e(e, v1, v2, graph):
    if v1 not in graph:
        print(v1, " not in graph")

    elif v2 not in graph:
        print(v2, " not in graph")
    else:
        temp = [v2, e]
        graph[v1].append(temp)


def print_graph(graph):
    for v in graph:
        for e in graph[v]:
            print(v, " -> ", e[0], " edge weight: ", e[1])


def check_cyclic(graph, n_nodes, vertex):
    visited = [False for i in range(n_nodes)]
    if visited[graph[vertex]]:
        return False


def makeNXgraph(g):
    G = nx.Graph()
    start = 1;
    for i in range(n_cities):
        for j in range(start, n_cities):
            if g[i][j] == 1.0:
                G.add_edge(i, j)
        start += 1

    return G


# Recursive function decomposes a graph into sub graphs

def network_reliability(G, graph, r, n_cities):
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
                    R1 = (1 - r[i][j]) * network_reliability(T, t_graph, r, n_cities)

                    # Second recursive call. Passes in original graph from this call, with the reliability at this index set to 1.
                    R2 = r[i][j] * network_reliability(G, graph, t_r, n_cities)

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


def find_best_edge(graph, n_cities, r, c):
    H = graph.copy()
    cycles = nx.cycle_basis(H, root=None)
    l = len(cycles)
    new_r = []
    start = 1

    for i in range(n_cities):
        for j in range(start, n_cities):
            r_ = r
            if graph[i][j] == 1.0:
                r_[i][j] = 2 * r_[i][j] / r_[i][j] ** 2

            new_r[i][j] = network_reliability(graph, r_)
        start += 1


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

    if len(cycles) == 0:
        noncycleEdges = graph.copy()

        for i in range(len(graph[0])):
            for j in range(len(graph[0])):
                if noncycleEdges[i][j] >= 1:
                    rels.append(r[i][j])
    else:
        for k in range(len(cycles)):
            for i in range(numcities):
                for j in range(numcities):
                    if i in cycles[k] and j in cycles[k] or graph[i][j] == 0:
                        noncycleEdges[i][j] = 1

        for i in range(len(graph[0])):
            for j in range(len(graph[0])):
                if noncycleEdges[i][j] == 0:
                    rels.append(r[i][j])

    return noncycleEdges, rels


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


def find_best(G, graph, n_cities, r, c):
    cycles = nx.cycle_basis(G, root=None)
    rel = get_cycles_reliability(cycles, r)

    print(rel)

    noncycleEdges, noncycleRels = get_edges_not_in_cycle(cycles, graph, r, n_cities)
    H = G.copy()
    t_graph = graph.copy()
    cycles = nx.cycle_basis(H, root=None)
    l = len(cycles)
    new_r = 0
    ratio = np.zeros((6, 6))
    best_ratio, besti, bestj, rest_rel = 0, 0, 0, 0

    start = 1
    r_old = network_reliability(H, t_graph, r, n_cities)
    print(r_old)
    #
    for i in range(n_cities):
        for j in range(start, n_cities):
            H = G.copy()
            t_graph = graph.copy()
            r_ = r.copy()
            if t_graph[i][j] >= 1.0:  # If adding parallel edge to already present edge
                r_[i][j] = 2 * r_[i][j] / r_[i][j] ** 2
            else:
                H.add_edge(i, j)
                t_graph[i][j] += 1

            new_r = network_reliability(H, t_graph, r_, n_cities)
            ratio[i][j] = new_r - r_old / c[i][j]
            if ratio[i][j] > best_ratio:
                best_ratio = ratio[i][j]
                besti, bestj, best_rel = i, j, new_r
        start += 1

    # print(besti)
    # print(bestj)
    print(best_ratio)
    # print(new_r)

    return besti, bestj, new_r


def connect(n_cities, rels, costs, visited):
    graph = np.zeros((6, 6))

    ind = np.dstack(np.unravel_index(np.argsort((-rels).ravel()), (6, 6)))
    ind = ind[0]

    edge_count = 0

    for edge in range(18):
        if visited[ind[edge][0]] == True and visited[ind[edge][1]] == True:
            continue
        elif edge_count == 5:
            break
        else:
            graph[ind[edge][0]][ind[edge][1]] = 1;
            # graph[ind[edge][1]][ind[edge][0]] = 1;
            edge_count += 1
            visited[ind[edge][0]] = True
            visited[ind[edge][1]] = True

    print(graph)

    total_reliability = 1
    total_cost = 0
    for i in range(n_cities):
        for j in range(n_cities):
            if graph[i][j] == 1.0:
                total_reliability *= rels[i][j]
                total_cost += costs[i][j]

    return graph, total_reliability, total_cost


# print(cost_mat)
# print(rel_mat)
#
# visited = [False for i in range(n_cities)]
#
# g, r, c = connect(0.9, 50, n_cities, rel_mat, cost_mat, visited)
# G = makeNXgraph(g)
#
#
# G.add_edge(0, 5)
# g[0][5] = 1
#
# find_best(G, g, n_cities, rel_mat, cost_mat)

# G = nx.Graph()
# G.add_edges_from([[1,2],[2,3],[3,1],[4,5]])
#
# components = []
# for graph in nx.connected_components(G):
#   components.append([graph, len(graph)])
#
# print(components)


# def get_edges_not_in_cycle(cycles, graph, r, numcities):
# 	noncycleEdges = np.ones((6,6))
# 	for k in range(len(cycles)):
# 		for i in range(numcities):
# 			for j in range(numcities):
# 				if i in cycles[k] and j in cycles[k] or graph[i][j]==0:
# 					noncycleEdges[i][j]=0
#
# 	rels = []
# 	for i in range(len(graph[0])):
# 		for j in range(len(graph[0])):
# 			if noncycleEdges[i][j]==1:
# 				rels.append(r[i][j])


def get_edges_from_adj_matrix(am):
    edges = []
    for i in range(n_cities):
        for j in range(n_cities):
            if am[i, j] == 1:
                edges.append([i, j])
    return edges

def calc_cost(am):
    cost = 0
    for i in range(n_cities):
        for j in range(n_cities):
            if am[i, j] > 0:
                cost += am[i, j] * cost_mat[i, j]
    return cost


def driver(r_target, cost_target):
    visited = [False for i in range(n_cities)]
    # create mst
    adj_matrix, mst_reliability, mst_cost = connect(n_cities, rel_mat, cost_mat, visited)

    if mst_cost > cost_target:
        print(f"Cities cannot be connected for cost {cost_target}")
        return
    if mst_reliability >= r_target:
        # Have a function to draw graph of network here
        print("cities can be connected with target reliablility and cost")

    # create nx graph to take advantage of built-in functionality
    graph_nx = nx.Graph()

    # create an add edge function to iteratively try to add new edges and test if meets reliability/cost goal
    ''' 
    possible approach:
    search for nodes with no edge between them
    attempt to 
    '''

    total_R = 1
    n_R = 0
    total_C = calc_cost(adj_matrix)
    edges = get_edges_from_adj_matrix(adj_matrix)
    graph_nx.add_edges_from(edges)

    # cycles = nx.cycle_basis(graph_nx, root=None)
    # cycles_reliability = get_cycles_reliability(cycles, rel_mat)
    while total_C < cost_target and n_R < r_target:
        best_i, best_j, n_R = find_best(graph_nx, adj_matrix, n_cities, rel_mat, cost_mat)
        print(best_i, best_j)
        adj_matrix[best_i, best_j] += 1.0

        if adj_matrix[best_i, best_j] == 1:
            graph_nx.add_edge(best_i, best_j)

        total_C = calc_cost(adj_matrix)
        total_R *= n_R

    print(n_R)

    # get the non-cycle edges and their respective reliabilities
    # non_cycle_edges, nce_rels = get_edges_not_in_cycle(cycles, adj_matrix, rel_mat, n_cities)
    # non_cycle_edges = get_edges_from_adj_matrix(non_cycle_edges)

    # print(total_R)

driver(0.9, 150)
