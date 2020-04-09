################################################################################
## NeuralTDA: Routines for analyzing population spike trains with Topology #####
## Brad Theilman 2019-09-16 											   #####
################################################################################

import os
import subprocess

from collections import Counter
from itertools import repeat, combinations, groupby
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.special import comb
import scipy.sparse.linalg as spla
from scipy.interpolate import interp1d

import h5py as h5

################################################################################
## Spike Pipeline ##############################################################
################################################################################
def kwik_get_clusters(kwikfile):
    ''' 
    Get all the trials from a sorted, merged kwikfile
    Trials are returned as a list with tuples for each trial: 
        (stim_name, stim_start_time, stim_end_time)
    All times are in samples relative to start of kwikfile (block)
    '''

    with h5.File(kwikfile, 'r') as f:
        cluster_ids = list(f['/channel_groups/0/clusters/main/'].keys())
        cluster_ids = sorted([int(x) for x in cluster_ids])
        cluster_groups = [f['/channel_groups/0/clusters/main/{}'.format(x)].attrs.get('cluster_group') for x in cluster_ids]
    return zip(cluster_ids, cluster_groups)

def kwik_get_trials(kwikfile):
    ''' 
    Get all the trials from a sorted, merged kwikfile
    Trials are returned as a list with tuples for each trial: 
        (stim_name, stim_start_time, stim_end_time)
    All times are in samples relative to start of kwikfile (block)
    '''

    with h5.File(kwikfile, 'r') as f:
        stim_names = list(f['/event_types/Stimulus/text'])
        stim_names = [x.decode('utf-8') for x in stim_names]
        stim_start_times = list(f['/event_types/Stimulus/time_samples'])
        stim_end_times = list(f['/event_types/Stimulus/stimulus_end'])
    return list(zip(stim_names, stim_start_times, stim_end_times))
        
    
def kwik_get_spikes(kwikfile):
    '''
    Get all the spikes froma sorted, merged kwikfile
    Spikes are returned as an Nspike x 2 numpy array
    spikes[:, 0] is spike time in samples relative to kwikfile (block) start
    spikes[:, 1] is cluster id
    '''
    
    with h5.File(kwikfile, 'r') as f:
        spikes_clus = np.array(f['/channel_groups/0/spikes/clusters/main'])
        spikes_times = np.array(f['/channel_groups/0/spikes/time_samples'])
        
    spikes = np.vstack((spikes_times, spikes_clus)).T
    return spikes

def spikes_in_interval(spikes, t_lo, t_hi, cell_group):
    """ 
    Given an array of spikes, find all the spikes whose time is between t_lo and t_hi inclusive
    store them in the 
    """
    if len(spikes) == 0:
        return

    m = int(np.floor((len(spikes)) / 2))

    if t_lo < spikes[m, 0]:
        spikes_in_interval(spikes[0:m], t_lo, t_hi, cell_group)
    if t_lo <= spikes[m, 0] and t_hi >= spikes[m, 0]:
        cell_group.append( (spikes[m, 0], spikes[m, 1]))
    if t_hi > spikes[m, 0]:
        spikes_in_interval(spikes[m + 1 :], t_lo, t_hi, cell_group)


def get_spikes_in_interval(spikes, t_lo, t_hi):
    cg = []
    spikes_in_interval(spikes, t_lo, t_hi, cg)
    return cg

def get_trial_spiketrains(spikes, trials, padding_secs, fs):
    ''' Organize all trials into a dictionary:
    trial_spiketrains[stim] = list of spiketrains
    '''
    trial_spiketrains = {}
    for trial in trials:
        
        stim_name = trial[0]
        stim_start = trial[1]
        stim_end = trial[2]
        
        padding_samps = np.round(padding_secs*fs)
        trial_start = np.amax([0, stim_start - padding_samps])
        trial_end = stim_end + padding_samps
        
        trial_spikes = []
        tp3.spikes_in_interval(spikes, trial_start, trial_end, trial_spikes)
        if stim_name not in trial_spiketrains.keys():
            trial_spiketrains[stim_name] = []
        trial_spiketrains[stim_name].append((trial_start, trial_end, stim_start, stim_end, trial_spikes))
    return trial_spiketrains

def spike_id(spike):
    return spike[1]

def spike_time(spike):
    return spike[0]

def get_unit_spike_times(spikes):
    sorted_spikes = sorted(spikes, key=spike_id)
    spiketimes = []
    units = []
    for k, g in groupby(sorted_spikes, spike_id):
        unit_spiketimes = [x[0] for x in g]
        spiketimes.append(unit_spiketimes)
        units.append(k)
    return (units, spiketimes)

################################################################################
## Betti Number Pipeline #######################################################
################################################################################


def get_windows(t_start, t_end, win_len, skip):
    """
    Given a time interval [t_start, t_end], 
    compute lists of window start and end times.
    Windows start every 'skip' samples and are 'win_len' samples long
    -checked
    """
    assert skip > 0
    win_starts = np.arange(t_start, t_end, skip)
    win_ends = win_starts + win_len - 1
    return (win_starts, win_ends)

def total_firing_rates(spikes, stim_start, stim_end):
    
    stim_spikes = []
    spikes_in_interval(spikes, stim_start, stim_end, stim_spikes)
    spike_ids = [x[1] for x in stim_spikes]
    c = Counter(spike_ids)
    return c
    
def spike_list_to_cell_group(spike_list, clu_rates, thresh, dt, T):
    """ 
    Given a spike list, first counts number of each spikes from each cluster
    then given a dictionary that maps clusters to total firing rates, computes whether the cluster's firing
    rate in the spike list exceeded some threshold by dividing the cluster's number of in-window spikes by the 
    window length in seconds and then checking if that value is greater than the threshold times the cluster total firing rate givne in clu_rates
    """
    spike_ids = [x[1] for x in spike_list]
    c = Counter(spike_ids)
    cg = set()

    for clu in clu_rates.keys():
        if (c[clu] / dt) >= (thresh * clu_rates[clu] / T):
            cg.add(clu)
    return cg

def spikes_to_cell_groups(spikes, stim_start, stim_end, win_len, fs, thresh):
    
    total_frs = total_firing_rates(spikes, stim_start, stim_end)
    win_starts, win_ends = get_windows(stim_start, stim_end, win_len, win_len)
    cell_groups = []
    dt = win_len / fs
    T = (stim_end - stim_start) / fs
    for ind, (ws, we) in enumerate(zip(win_starts, win_ends)):
        spike_list = []
        spikes_in_interval(spikes, ws, we, spike_list)
        if spike_list:
            window_time = (we + ws) / 2 - stim_start
            cell_groups.append((ind, window_time, sorted(tuple(spike_list_to_cell_group(spike_list, total_frs, dt, thresh, T)))))
    return cell_groups

def cell_groups_to_bin_mat(cell_groups, clus, nwins):
    ncells = len(clus)
    bin_mat = np.zeros((ncells, nwins))
    for cg in cell_groups:
        cginds = np.isin(clus, cg[2])
        bin_mat[cginds, cg[0]] = 1
    return bin_mat

def build_perseus_persistent_input(cell_groups, savefile):
    """
    Formats cell group information as an input file
    for the Perseus persistent homology software, but assigns filtration
    levels for each cell group based on the time order of their appearance
    in the signal.

    Parameters
    ----------
    cell_groups : list
        cell_group information returned by spikes_to_cell_groups
    savefile : str
        File in which to put the formatted cellgroup information

    Yields
    ------
    savefile : text File
        file suitable for running perseus on
    """
    with open(savefile, "w+") as pfile:
        pfile.write("1\n")
        for cell_group in cell_groups:
            grp = list(cell_group[2])
            grp_dim = len(grp) - 1
            if grp_dim < 0:
                continue
            vert_str = str(grp)
            vert_str = vert_str.replace("[", "")
            vert_str = vert_str.replace("]", "")
            vert_str = vert_str.replace(" ", "")
            vert_str = vert_str.replace(",", " ")
            out_str = (
                str(grp_dim) + " " + vert_str + " {}\n".format(str(cell_group[0] + 1))
            )
            pfile.write(out_str)
    return savefile

def run_perseus(pfile):
    """
    Runs Perseus persistent homology software on the data in pfile

    Parameters
    ------
    pfile : str
        File on which to compute homology

    Returns
    ------
    betti_file : str
        File containing resultant betti numbers

    """
    pfile_split = os.path.splitext(pfile)
    of_string = pfile_split[0]
    perseus_command = "perseus"
    perseus_return_code = subprocess.call(
        [perseus_command, "nmfsimtop", pfile, of_string]
    )

    betti_file = of_string + "_betti.txt"
    # betti_file = os.path.join(os.path.split(pfile)[0], betti_file)

    return betti_file

def read_perseus_result(betti_file):
    bettis = []
    f_time = []
    maxbetti = 10
    try:
        with open(betti_file, "r") as bf:
            for bf_line in bf:
                if len(bf_line) < 2:
                    continue
                betti_data = bf_line.split()
                filtration_time = int(betti_data[0])
                betti_numbers = list(map(int, betti_data[1:]))
                betti_numbers = (betti_numbers + maxbetti*[0])[:maxbetti]
                bettis.append([filtration_time, betti_numbers])
    except:
        bettis.append([-1, [-1]])
    return bettis

def compute_bettis(spikes, stim_start, stim_end, win_len, fs, thresh):
    ''' win_len in samples'''
    win_starts, win_ends = get_windows(stim_start, stim_end, win_len, win_len)
    cell_groups = spikes_to_cell_groups(spikes, stim_start, stim_end, win_len, fs, thresh)
    build_perseus_persistent_input(cell_groups, './test.betti')
    betti_file = run_perseus('./test.betti')
    betti_nums = read_perseus_result(betti_file)
    betti_nums = [[win_starts[x[0]-1] - stim_start, x[0], x[1]] for x in betti_nums]
    
    return betti_nums
    
def betti_curve_func(betti_nums, dim, stim_start, stim_end, fs, t_in_seconds=False):
    
    betti_ts = [x[0] for x in betti_nums]
    betti_vals = [x[2][dim] for x in betti_nums]
    if t_in_seconds:
        betti_ts = list(map(lambda x: x / fs, betti_ts))
    f = interp1d(betti_ts, betti_vals, kind='zero', bounds_error = False, fill_value=(0, betti_vals[-1]))
    return f

def shuffle_trial_spikes(spikes, stim_start, stim_end):
    # trial_spikes is a list of (sample, id) spikes
    # for each spike, generate a new sample in range stim_start, stim_end
    trial_spikes = []
    tp3.spikes_in_interval(spikes, stim_start, stim_end, trial_spikes)
    new_spikes_times = []
    new_spikes_ids = []
    for spike in trial_spikes:
        new_sample = np.random.randint(stim_start, high=stim_end+1)
        new_spikes_times.append(new_sample)
        new_spikes_ids.append(spike[1])
    out_spikes = np.vstack((new_spikes_times, new_spikes_ids)).T
    out_spikes_ind = np.argsort(out_spikes[:, 0])
    
    return out_spikes[out_spikes_ind, :]

def get_betti_curves(spikes, stim_start, stim_end, win_len, fs, thresh, maxdim):

    t = np.linspace(0, stim_end - stim_start, 2000) / fs
    betti_nums = compute_bettis(spikes, stim_start, stim_end, win_len, fs, thresh=4.0)

    betti_curves_dims = {}
    for dim in range(maxdim):
        betti_func = betti_curve_func(betti_nums, dim, stim_start, stim_end, fs, t_in_seconds=True)
        betti_curve = betti_func(t)
        betti_curves_dims[dim] = betti_curve
    return (t, betti_curves_dims)
  


################################################################################
## Simplicial Laplacian Pipeline ###############################################
################################################################################


@lru_cache(maxsize=128)
def cached_comb(n, k):
    return comb(n, k, exact=True)


def simplex_to_integer(simplex):
    # this assumes simplex in ascending order
    N = len(simplex)
    res = 0
    for p, n in enumerate(reversed(simplex)):
        res += cached_comb(n, N - p)
    return res


def boundary_op_dim(simplex, dim, boundaries):

    # Ensure simplex vertices are in ascending order
    simplex = sorted(simplex)

    # The ith boundary operator maps simpleces
    # with dim + 1 vertices to simplices with dim vertices
    nverts = dim + 1

    # Each dim subface of a max simplex is a dim+1 combination of the vertices
    for face in combinations(simplex, nverts):

        # compute the combinatoric number associated to the dim face
        simp_num = simplex_to_integer(face)

        # Compute the column of the boundary operator
        # by going through faces of dim subface
        for i in range(nverts):
            sgn = 1 - 2 * (i % 2)
            face2 = face[0:i] + face[i + 1 :]
            face_num = simplex_to_integer(face2)
            boundaries[dim].add((face_num, simp_num, sgn))


def boundaries_to_matrices(boundaries):
    bdry_mats = {}
    for dim in boundaries.keys():
        if len(boundaries[dim]) > 0:
            bdry = np.vstack(list(boundaries[dim]))
            bdry_mat = sp.coo_matrix((bdry[:, 2], (bdry[:, 0], bdry[:, 1])))
            bdry_mats[dim] = bdry_mat.tocsr()
    return bdry_mats


def laplacian(mats, dim):
    m1 = mats[dim]
    m2 = mats[dim + 1]
    sp1 = mats[dim].shape
    sp2 = mats[dim + 1].shape
    x = max(sp1[1], sp2[0])
    m1.resize((sp1[0], x))
    m2.resize((x, sp2[1]))
    return m1.T.dot(m1) + m2.dot(m2.T)


def spikes_to_laplacian(spikes, windows, dim):
    boundaries = {}
    for i in range(20):
        boundaries[i] = set()
    for win in windows:
        cg = set()
        spikes_in_interval(spikes, win[0], win[1], cg)
        boundary_op_dim(list(cg), dim, boundaries)
        boundary_op_dim(list(cg), dim + 1, boundaries)
    mats = boundaries_to_matrices(boundaries)
    return laplacian(mats, dim)

def spectrum_KL(r, s):
    # compute "KL"
    div = 0.0
    for rval,sval in zip(r, s):
        if rval < 1e-16 or sval < 1e-16:
            div += 0
        else:
            div += rval * (np.log(rval) - np.log(sval)) / np.log(2.0)
    return div

def slsa_compare_spiketrains_KL(spikes1, spikes2, windows, dim, beta):
    
    L1 = spikes_to_laplacian(spikes1, windows, dim)
    L2 = spikes_to_laplacian(spikes2, windows, dim)
    
    L1 = L1.tocsr()
    L2 = L2.tocsr()
    
    #compress matrix
    L1 = L1[L1.getnnz(1)!=0][:, L1.getnnz(0)!=0]
    L2 = L2[L2.getnnz(1)!=0][:, L2.getnnz(0)!=0]
    
    # pad to same dimensions
    L1s = L1.shape
    L2s = L2.shape
    x = max(L1s[0], L2s[0])
    L1.resize((x, x))
    L2.resize((x, x))
    
    # Get spectra
    spec_L1 = np.array(sorted(spla.svds(L1.astype(float), k=(min(L1.shape) - 1), return_singular_vectors = False)))
    spec_L2 = np.array(sorted(spla.svds(L2.astype(float), k=(min(L2.shape) - 1), return_singular_vectors = False)))
    
    # compute normalized exponential spectra
    r = np.exp(-beta*spec_L1)
    s = np.exp(-beta*spec_L2)
    
    r = np.array(sorted(r)) / np.sum(r)
    s = np.array(sorted(s)) / np.sum(s)
    
    # compute "KL"
    return spectrum_KL(r, s)
    
    
def slsa_compare_spiketrains_JS(spikes1, spikes2, windows, dim, beta):
    
    L1 = spikes_to_laplacian(spikes1, windows, dim)
    L2 = spikes_to_laplacian(spikes2, windows, dim)
    
    L1 = L1.tocsr()
    L2 = L2.tocsr()
    
    #compress matrix
    L1 = L1[L1.getnnz(1)!=0][:, L1.getnnz(0)!=0]
    L2 = L2[L2.getnnz(1)!=0][:, L2.getnnz(0)!=0]
    
    # pad to same dimensions
    L1s = L1.shape
    L2s = L2.shape
    x = max(L1s[0], L2s[0])
    L1.resize((x, x))
    L2.resize((x, x))
    
    M = (L1 + L2) / 2.0
    
    # Get spectra
    spec_L1 = np.array(sorted(spla.svds(L1.astype(float), k=(min(L1.shape) - 1), return_singular_vectors = False)))
    spec_L2 = np.array(sorted(spla.svds(L2.astype(float), k=(min(L2.shape) - 1), return_singular_vectors = False)))
    spec_M = np.array(sorted(spla.svds(M.astype(float), k=(min(M.shape) - 1), return_singular_vectors = False)))
    
    # compute normalized exponential spectra
    r = np.exp(-beta*spec_L1)
    s = np.exp(-beta*spec_L2)
    m = np.exp(-beta*spec_M)
    
    r = np.array(sorted(r)) / np.sum(r)
    s = np.array(sorted(s)) / np.sum(s)
    m = np.array(sorted(m)) / np.sum(m)
    
    # compute "KL"
    js = (spectrum_KL(r, m) + spectrum_KL(s, m)) / 2.0
    return js
