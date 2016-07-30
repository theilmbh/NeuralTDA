import networkx as nx 
import itertools
import topology
import numpy as np 

def build_graph_recursive(graph, cell_group, parent_name,):

	cell_group_name = ''.join(cell_group)
	graph.add_node(cell_group_name)
	if parent_name is not '':
		graph.add_edge(cell_group_name, parent_name)
		graph.edge[cell_group_name][parent_name]['name'] = cell_group_name+parent_name
	n_cells_in_group = len(cell_group)
	if n_cells_in_group > 1:
		for subgrp in itertools.combinations(cell_group, n_cells_in_group-1):
			build_graph_recursive(graph, subgrp, cell_group_name)

	return graph


def build_graph_from_cell_groups(cell_groups):

	graph = nx.Graph()
	for win, group in cell_groups:
		group_s = [str(s) for s in group]
		graph = build_graph_recursive(graph, group_s, '')
	return graph

def build_graph_from_binned_dataset(binned_dataset, thresh):

	cell_groups = topology.calc_cell_groups_from_binned_data(binned_dataset, thresh)
	graph = build_graph_from_cell_groups(cell_groups)
	return graph 

def build_graph_from_cell_groups_incremental(cell_groups, t):
	graph = nx.Graph()
	for win, group in cell_groups[:t]:
		group_s = [str(s) for s in group]
		graph = build_graph_recursive(graph, group_s, '')
	return graph

def compute_gamma_q(graph):

	gamma_q = nx.Graph()
	for (u,v, e_name) in graph.edges(data='name'):
		gamma_q.add_node(e_name)
		for (u_new, v_new, e_new_name) in graph.edges([u, v], data='name'):
			if (u_new != u) or (v_new != v):
				gamma_q.add_edge(e_name, e_new_name)
	return gamma_q 

def compute_ideal_generators(gamma_q):

	generators = []

	# Relation 1: x^2 = x
	for vert in gamma_q.nodes():
		gen_str = '{}^2 - {}'.format(vert)
		generators.append(gen_str)

	# Relation 2: x*y = 0 if no edge between x, y
	for (ned1, ned2) in nx.non_edges(gamma_q):
		gen_str = '{}*{}'.format(ned1, ned2)
		generators.append(gen_str)

	# Relation 3: x*y*x = x if edge between x, y
	for (ed1, ed2) in gamma_q.edges():
		gen_str1 = '{}*{}*{} - {}'.format(ed1, ed2, ed1, ed1)
		gen_str2 = '{}*{}*{} - {}'.format(ed2, ed1, ed2, ed2)
		generators.append(gen_str1)
		generators.append(gen_str2)

	# Relation 4: 