################################################################################
## NeuralTDA: Routines for analyzing population spike trains with Topology #####
## Brad Theilman 2019-09-16 											   #####
################################################################################

import os
import subprocess

from collections import Counter
from itertools import repeat, combinations
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.special import comb
import scipy.sparse.linalg as spla

################################################################################
## Spike Pipeline ##############################################################
################################################################################


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
        cell_group.append(spikes[m, 1])
    if t_hi > spikes[m, 0]:
        spikes_in_interval(spikes[m + 1 :], t_lo, t_hi, cell_group)


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
    c = Counter(stim_spikes)
    return c


def spike_list_to_cell_group(spike_list, clu_rates, thresh, dt, T):
    """ 
    Given a spike list, first counts number of each spikes from each cluster
    then given a dictionary that maps clusters to total firing rates, computes whether the cluster's firing
    rate in the spike list exceeded some threshold by dividing the cluster's number of in-window spikes by the 
    window length in seconds and then checking if that value is greater than the threshold times the cluster total firing rate givne in clu_rates
    """
    c = Counter(spike_list)
    cg = set()

    for clu in clu_rates.keys():
        if (c[clu] / dt) >= (thresh * clu_rates[clu] / T):
            cg.add(clu)
    return cg


def spikes_to_cell_groups(spikes, stim_start, stim_end, win_len, fs):

    total_frs = total_firing_rates(spikes, stim_start, stim_end)
    win_starts, win_ends = get_windows(stim_start, stim_end, win_len, win_len)
    cell_groups = []
    for ind, (ws, we) in enumerate(zip(win_starts, win_ends)):
        spike_list = []
        spikes_in_interval(spikes, ws, we, spike_list)
        if spike_list:
            window_time = (we + ws) / 2 - stim_start
            cell_groups.append(
                (
                    ind,
                    window_time,
                    sorted(
                        tuple(spike_list_to_cell_group(spike_list, total_frs, dt, 1, T))
                    ),
                )
            )
    return cell_groups


def cell_groups_to_bin_mat(cell_groups, ncells, nwins):

    bin_mat = np.zeros((ncells, nwins))
    for cg in cell_groups:
        bin_mat[cg[2], cg[0]] = 1
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
    try:
        with open(betti_file, "r") as bf:
            for bf_line in bf:
                if len(bf_line) < 2:
                    continue
                betti_data = bf_line.split()
                filtration_time = int(betti_data[0])
                betti_numbers = list(map(int, betti_data[1:]))
                bettis.append([filtration_time, betti_numbers])
    except:
        bettis.append([-1, [-1]])
    return bettis


def compute_bettis(spikes, stim_start, stim_end, win_len, fs):

    win_starts, win_ends = get_windows(stim_start, stim_end, win_len, win_len)
    cell_groups = spikes_to_cell_groups(spikes, stim_start, stim_end, win_len, fs)
    build_perseus_persistent_input(cell_groups, "./test.betti")
    betti_file = run_perseus("./test.betti")
    betti_nums = read_perseus_result(betti_file)

    betti_nums = [[win_starts[x[0]] - stim_start, x[0], x[1]] for x in betti_nums]
    return betti_nums


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