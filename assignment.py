
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

# rel_mat = np.zeros((6,6))
# cost_mat = np.zeros((6,6))
relMap = []
costMap = []

for i in range(n_cities):
	for j in range(start, n_cities):
		# rel_mat[i][j] = reliabilities[relcount]
		relMap.append((i,j,reliabilities[relcount]))
		# print(rel_mat[i][j])
		relcount += 1
	start += 1

relcount = 0
start = 1

for i in range(n_cities):
	for j in range(start, n_cities):
		# cost_mat[i][j] = costs[relcount]
		# print(cost_mat[i][j])
		costMap.append((i,j,costs[relcount]))
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
	if visited[graph[v]]:
		return False

def connect_cities(r, cost, n_cities, rels, costs):
	graph = {}

	rels.sort(key=lambda tup: tup[2], reverse=True)
	costs.sort(key=lambda tup: tup[2], reverse=True)
	# print(rels)

	for i in range(n_cities):
		add_v(i, graph)

	edges = []
	n_edges = n_cities - 1
	edge_count = 0
	nodes = [0 for i in range(n_cities)]

	for edge in rels:
		if nodes[edge[0]] > 0 and nodes[edge[1]] > 0:
			continue
		elif edge_count == 5:
			break
		else:
			edges.append(edge)
			edge_count+=1
			nodes[edge[0]] += 1
			nodes[edge[1]] += 1

	print(edges)



	print_graph(graph)

connect_cities(0.9, 50, n_cities, relMap, costMap)


