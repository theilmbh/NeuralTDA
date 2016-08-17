import networkx as nx 
import itertools
import topology
import numpy as np 
import h5py as h5 
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_graph_recursive(graph, cell_group, parent_name):

    cell_group_name = ''.join(cell_group)
    graph.add_node(cell_group_name)
    n_cells_in_group = len(cell_group)

    graph.add_edge(cell_group_name, parent_name)
    graph.edge[cell_group_name][parent_name]['name'] = cell_group_name+parent_name
    
    if n_cells_in_group > 1:
        for subgrp in itertools.combinations(cell_group, n_cells_in_group-1):
            build_graph_recursive(graph, subgrp, cell_group_name)

    return graph

def build_metric_graph_recursive(graph, cell_group, parent_name, Ncells, tau):

    cell_group_name = ''.join(cell_group)
    graph.add_node(cell_group_name)
    n_cells_in_group = len(cell_group)
    muk = 1.0
    muk = 1.0-np.pi*np.sqrt(float(n_cells_in_group-1)/float(Ncells))
    if n_cells_in_group==0:
        muk=1.0
    muk = np.exp(-tau*n_cells_in_group/float(Ncells))
    graph.add_edge(cell_group_name, parent_name, weight=muk)
    graph.edge[cell_group_name][parent_name]['name'] = cell_group_name+parent_name
    graph.edge[cell_group_name][parent_name]['weight'] = muk
    if n_cells_in_group > 1:
        for subgrp in itertools.combinations(cell_group, n_cells_in_group-1):
            build_metric_graph_recursive(graph, subgrp, cell_group_name, Ncells, tau)

    return graph

def power_dist(ncells1, ncells2, tau, c):

    return c*np.power(np.abs(ncells1 - ncells2)/ncells2, tau)

def build_arb_metric_graph_recursive(graph, cell_group, parent_name, distance_func):

    cell_group_name = ''.join(cell_group)
    graph.add_node(cell_group_name)
    n_cells_in_group = len(cell_group)
    muk = distance_func(n_cells_in_group, n_cells_in_group+1)
    graph.add_edge(cell_group_name, parent_name, weight=muk)
    graph.edge[cell_group_name][parent_name]['name'] = cell_group_name+parent_name
    graph.edge[cell_group_name][parent_name]['weight'] = muk
    if n_cells_in_group > 1:
        for subgrp in itertools.combinations(cell_group, n_cells_in_group-1):
            build_arb_metric_graph_recursive(graph, subgrp, cell_group_name, distance_func)

    return graph

def build_graph_from_cell_groups(cell_groups):

    graph = nx.Graph()
    prev=''
    for win, group in cell_groups:
        group_s = [str(s)+'-' for s in sorted(group)]
        cell_group_name = ''.join(group_s)
        graph = build_graph_recursive(graph, group_s, '')
        graph.add_edge(prev, cell_group_name)
        prev=cell_group_name

    return graph

def build_metric_graph_from_cell_groups(cell_groups, Ncells, tau):

    graph = nx.Graph()
    prev=''
    for win, group in cell_groups:
        group_s = [str(s)+'-' for s in sorted(group)]
        cell_group_name = ''.join(group_s)
        graph = build_metric_graph_recursive(graph, group_s, '', Ncells, tau)
        graph.add_edge(prev, cell_group_name, weight=1.0)
        prev=cell_group_name

    return graph

def build_arb_metric_graph_from_cell_groups(cell_groups, distance_func):

    graph = nx.Graph()
    prev=''
    for win, group in cell_groups:
        group_s = [str(s)+'-' for s in sorted(group)]
        cell_group_name = ''.join(group_s)
        graph = build_arb_metric_graph_recursive(graph, group_s, '', distance_func)
        graph.add_edge(prev, cell_group_name, weight=1.0)
        prev=cell_group_name

    return graph

def build_power_metric_graph_from_cell_groups(cell_groups, dfunc_params):

    tau = dfunc_params['tau']
    c = dfunc_params['c']
    dfun = lambda ncell1, ncell2: power_dist(ncell1, ncell2, tau, c)

    graph = build_arb_metric_graph_from_cell_groups(cell_groups, dfun)
    return graph

def build_graph_from_binned_dataset(binned_dataset, thresh):

    cell_groups = topology.calc_cell_groups_from_binned_data(binned_dataset, thresh)
    graph = build_graph_from_cell_groups(cell_groups)
    return graph 

def build_graph_from_cell_groups_incremental(cell_groups, t):
    graph = nx.Graph()
    for win, group in cell_groups[:t]:
        group_s = [str(s) for s in sorted(group)]
        graph = build_graph_recursive(graph, group_s, '')
    return graph

def get_cell_group_trajectory(cell_groups):
    vert_list = []
    for win, group in cell_groups:
        group_s = [str(s)+'-' for s in group]
        group_s = ''.join(group_s)
        vert_list.append(group_s)
    return vert_list

def concatenate_all_trial_cell_groups(binned_datafile, stimulus, thresh):

    cg_concat = []
    with h5.File(binned_datafile, 'r') as f:
        stimdata = f[stimulus]
        for trial in stimdata.keys():
            bds = stimdata[trial]
            cgs = topology.calc_cell_groups_from_binned_data(bds, thresh)
            cg_concat = cg_concat + cgs 
    return cg_concat

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

def do_HMDS(input_file, output_file, n, eps, eta, maxiter, maxtrial, verbose):

    HMDS_command = "/home/btheilma/bin/hmds"
    sbp_arg_list = [HMDS_command, '-i', input_file, '-o', output_file, '-n', str(n), '-e', str(eps), '-h', str(eta), '-m', str(maxiter)]
    if verbose:
        sbp_arg_list.append('-v')
    if maxtrial:
        sbp_arg_list.append('-t')
        sbp_arg_list.append(str(maxtrial))
    subprocess.call(sbp_arg_list)

def run_HMDS(distmat, hmds_params):

    infile = os.path.abspath('~/hmds_in.dat')
    outfile = os.path.abspath('~/hmds_out.dat')

    n = hmds_params['n']
    eps = hmds_params['eps']
    eta = hmds_params['eta']
    maxiter = hmds_params['maxiter']
    maxtrial = hmds_params['maxtrial']
    verbose = hmds_params['verbose']

    with open(infile, 'wb') as f:
        distmat.tofile(f)
    do_HMDS(infile, outfile, n, eps, eta, maxiter, maxtrial, verbose)

    with open(outfile, 'r') as f:
        hmds_out = np.fromfile(f)
    return hmds_out

def plot_pf_graph_recursive(binned_dataset, thresh, title, savepath):

    if 'pop_vec' in binned_dataset.keys():
        graph = build_graph_from_binned_dataset(binned_dataset, thresh)
        cgs = topology.calc_cell_groups_from_binned_data(binned_dataset, thresh)

        cg_traj = get_cell_group_trajectory(cgs)
        pos = nx.spectral_layout(graph)
        green_nodes = [(s in cg_traj) for s in graph.nodes()]
        green_edges = [(s, k) for s,k in zip(cg_traj[:-1], cg_traj[1:])]
        nodecolors = [('r' if s else 'g') for s in green_nodes]
        edgecolors = [('r' if (s in green_edges or s[::-1] in green_edges) else 'k') for s in graph.edges()]

        f = plt.figure(figsize=(22,22))
        nx.draw_networkx(graph, pos=pos, edge_color=edgecolors, node_color=nodecolors, node_size=50, with_labels=False)
        savepath = savepath +'.png'
        plt.savefig(savepath)
        plt.close(f)
    else:
        for num, ite in enumerate(binned_dataset.keys()):
            new_title = title+'-{}-'.format(ite)
            new_savepath = savepath+'-{}-'.format(ite)
            plot_pf_graph_recursive(binned_dataset[ite], thresh, new_title, new_savepath)

def make_pf_graph_plots(binned_datafile, thresh, savepath):

    with h5.File(binned_datafile, 'r') as bdf:

        stims = bdf.keys()
        for stim in stims:
            title = stim 
            new_savepath = savepath + title
            plot_pf_graph_recursive(bdf[stim], thresh, title, new_savepath)

def compute_pf_distance_matrix(graph):

    distances = nx.all_pairs_dijkstra_path_length(graph)
    nverts = graph.number_of_nodes()
    dmat = np.zeros((nverts, nverts))
    for n1, key1 in enumerate(graph.nodes()):
        dat = distances[key1]
        for n2, key2 in enumerate(graph.nodes()):
            dmat[n1, n2] = dat[key2]
    return dmat 

def plot_hyperbolic_embed(X, title, savefile):

    embed = X[0::2] + 1j*X[1::2]
    f = plt.figure(figsize=(22,22))
    plt.scatter(np.real(embed), np.imag(embed))
    plt.title(title)

    plt.savefig(savefile)
    plt.close(f)

def hyperbolic_embed_recursive(binned_dataset, thresh, title, savepath, dfunc_params, hmds_params):

    if 'pop_vec' in binned_dataset.keys():

        cgs = topology.calc_cell_groups_from_binned_data(binned_dataset, thresh)
        grph = build_power_metric_graph_from_cell_groups(cgs, c, tau)
        dmat = compute_pf_distance_matrix(grph)
        X = run_HMDS(dmat, hmds_params)
        plot_hyperbolic_embed(X, title, savepath+'.png')
    else:
        for num, ite in enumerate(binned_dataset.keys()):
            new_title = title+'-{}-'.format(ite)
            new_savepath = savepath+'-{}-'.format(ite)
            hyperbolic_embed_recursive(binned_dataset[ite], thresh, new_title, new_savepath, dfunc_params, hmds_params)

def make_hyperbolic_embeds(binned_datafile, thresh, savepath, hmds_params):

    with h5.File(binned_datafile, 'r') as bdf:
        stims = bdf.keys()
        for stim in stims:
            title = stim 
            new_savepath = savepath + title
            hyperbolic_embed_recursive(bdf[stim], thresh, title, new_savepath, dfunc_params, hmds_params)


