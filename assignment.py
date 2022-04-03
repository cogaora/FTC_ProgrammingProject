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

rel_mat = np.zeros((6,6))
cost_mat = np.zeros((6,6))
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

# def network_reliability(graph, r):


def find_best_edge(graph, n_cities, r, c):
	H = graph
	cycles = nx.cycle_basis(H, root=None)
	l = len(cycles)
	new_r = []
	start = 1

	for i in range(n_cities):
		for j in range(start, n_cities):
			r_ = r
			if graph[i][j] == 1.0:
				r_[i][j] = 2*r_[i][j]/r_[i][j]**2

			new_r[i][j] = network_reliability(graph, r_)
		start += 1



				# if l > 1:
				# 	H.remove_edge(i, j)
				# 	cycles = nx.cycle_basis(H, root=None)
				# 	if len(cycles) == l-1:
				# 		continue
				# 	else:
				# 		H.add_edge(i, j)

def get_edges(cycle, r):
	rels = np.zeros(len(cycle))
	for e in range(len(cycle)):
		if e == len(cycle)-1:
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

def get_edges_not_in_cycle(cycles, graph, r):
	noncycleEdges = np.ones((6,6))
	for k in range(len(cycles)):
		for i in range(len(graph[0])):
			for j in range(len(graph[0])):
				if i in cycles[k] and j in cycles[k] or graph[i][j]==0:
					noncycleEdges[i][j]=0

	rels = []
	for i in range(len(graph[0])):
		for j in range(len(graph[0])):
			if noncycleEdges[i][j]==1:
				rels.append(r[i][j])


	# print(noncycleEdges)
	return noncycleEdges, rels

def get_cycles_reliability(cycles, r):
	l = len(cycles)
	rel = np.zeros(l)
	for i in range(l):
		edges = get_edges(cycles[i], r)
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

def find_best(graph, n_cities, r, c):

	cycles = nx.cycle_basis(graph, root=None)
	rel = get_cycles_reliability(cycles, r)
	noncycleEdges, noncycleRels = get_edges_not_in_cycle(cycles,graph, r)
	print(noncycleEdges)
	print(noncycleRels)
	H = graph
	cycles = nx.cycle_basis(H, root=None)
	l = len(cycles)
	new_r = 0
	ratio = []
	best_ratio = 0
	start = 1
	r_old = network_reliability(graph, r)

	for i in range(n_cities):
		for j in range(start, n_cities):
			r_ = r
			if graph[i][j] == 1.0:
				r_[i][j] = 2 * r_[i][j] / r_[i][j] ** 2
			else:
				H.add_edge(i, j)

			new_r = network_reliability(graph, r_)
			ratio[i][j] = new_r-r_old/c[i][j]
			# if (ratio[i][j]>)
		start += 1



def connect(r, cost, n_cities, rels, costs, visited):
	graph = np.zeros((6,6))

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
			edge_count+=1
			visited[ind[edge][0]] = True
			visited[ind[edge][1]] = True

	print(graph)
	# print_graph(graph)

	total_reliability = 1
	total_cost = 0
	for i in range(n_cities):
		for j in range(n_cities):
			if graph[i][j] == 1.0:
				total_reliability *= rels[i][j]
				total_cost += costs[i][j]

	return graph, total_reliability, total_cost;


print(cost_mat)
print(rel_mat)

visited = [False for i in range(n_cities)]

g, r, c = connect(0.9, 50, n_cities, rel_mat, cost_mat, visited)

G = nx.Graph()
for i in range(n_cities):
	for j in range(n_cities):
		if g[i][j] == 1.0:
			G.add_edge(i, j)

G.add_edge(0, 5)

print(g)
print(G)
print(r)
print(c)

# find_best(G, n_cities, rel_mat, cost_mat)

# G = nx.Graph()
# G.add_edges_from([[1,2],[2,3],[3,1],[4,5]])
#
# components = []
# for graph in nx.connected_components(G):
#   components.append([graph, len(graph)])
#
# print(components)

