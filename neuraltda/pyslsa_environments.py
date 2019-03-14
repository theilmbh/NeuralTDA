import numpy as np
import pycuslsa as pyslsa
import matplotlib.pyplot as plt
import neuraltda.topology2 as tp2
import neuraltda.stimulus_space as ss
import pandas as pd
import h5py as h5
import pickle
import tempfile

from scipy.optimize import fmin

import tqdm

import os
import datetime

daystr = datetime.datetime.now().strftime("%Y%m%d")
figsavepth = "/home/brad/DailyLog/" + daystr + "/"
print(figsavepth)

# Class to define environments with holes
class TPEnv:
    def __init__(self, n_holes, hole_rad):
        self.xlim = [-1, 1]
        self.ylim = [-1, 1]
        self.holes = []
        self.hole_rad = hole_rad
        c = 0.75 * (2 * np.random.rand(2) - 1)
        for hole in range(n_holes):
            while self.hole_collide(c):
                c = 0.75 * (2 * np.random.rand(2) - 1)
            self.holes.append(c)
        # self.holes = 0.75*(2*np.random.rand(n_holes, 2) - 1) # keep centers in range -1, 1
        self.hole_rad = hole_rad  # radius of holes

    def in_hole(self, x, y):
        """
        Check to see if a point is in a hole
        """
        for hole in self.holes:
            if np.linalg.norm(np.subtract([x, y], hole)) < self.hole_rad:
                return True
        return False

    def hole_collide(self, c):
        """
        Check to see if a hole will collide with already existing holes 
        """

        for h in self.holes:
            if np.sqrt((h[0] - c[0]) ** 2 + (h[1] - c[1]) ** 2) <= 2 * self.hole_rad:
                return True
        return False


def generate_environments(N, h, numrepeats=1):
    envs = []
    for nholes in range(N):
        for r in range(numrepeats):
            envs.append(TPEnv(nholes, h))
    return envs


def convert_env_to_img(env, NSQ):
    img = np.ones((NSQ, NSQ))
    X, Y = np.meshgrid(np.linspace(-1, 1, NSQ), np.linspace(-1, 1, NSQ))

    for hole in env.holes:
        hx = hole[0]
        hy = hole[1]
        diffx = X - hx * np.ones(np.shape(X))
        diffy = Y - hy * np.ones(np.shape(Y))
        dists = np.sqrt(np.power(diffx, 2) + np.power(diffy, 2))
        img[dists < env.hole_rad] = 0

    return img


def compute_env_img_correlations(imgs):
    nsq, _ = np.shape(imgs[0])
    dat_mat = np.zeros((len(imgs), nsq * nsq))
    for ind, img in enumerate(imgs):
        dat_mat[ind, :] = img.flatten()

    cormat = np.corrcoef(dat_mat)
    return cormat


def generate_paths(space, n_steps, ntrials, dl):
    # pick a starting point
    final_pts = np.zeros((ntrials, n_steps, 2))
    for trial in range(ntrials):
        pts = []
        pt = (2 * np.random.rand(1, 2) - 1)[0]
        while space.in_hole(pt[0], pt[1]):
            pt = (2 * np.random.rand(1, 2) - 1)[0]
        # pts.append(pt)
        steps_to_go = n_steps
        while steps_to_go > 0:
            if steps_to_go % 10000 == 0:
                print("Steps to go: {}".format(steps_to_go))
            # pick a new point
            # theta = np.pi*np.random.rand(1)[0] - np.pi/2

            theta = 2 * np.pi * np.random.rand(1)[0]
            dx = dl * np.cos(theta)
            dy = dl * np.sin(theta)

            if (
                abs(pt[0] + dx) < 1
                and abs(pt[1] + dy) < 1
                and not space.in_hole(pt[0] + dx, pt[1] + dy)
            ):

                steps_to_go -= 1

                pt[0] = pt[0] + dx
                pt[1] = pt[1] + dy
                pts.append(np.copy(pt))
        pts = np.vstack(pts)
        final_pts[trial, :, :] = pts
    return final_pts


def generate_place_fields_random(n_fields, rad):

    centers = 2 * np.random.rand(n_fields, 2) - 1
    return (centers, rad)


def generate_place_fields(n_fields, rad):

    nf = np.round(np.sqrt(n_fields))
    cx = np.linspace(-1, 1, nf)
    cy = np.linspace(-1, 1, nf)
    centers = np.array([np.array((x, y)) for x in cx for y in cy])
    rads = rad * np.ones(n_fields)
    return (centers, rads)


def generate_place_fields_perturbed_lattice(n_fields, rad, stddev=0.1):
    """ generates place fields with centers normally perturbed around a lattice"""
    nf = np.round(np.sqrt(n_fields))
    cx = np.linspace(-1, 1, nf)
    cy = np.linspace(-1, 1, nf)
    cx = cx + stddev * np.random.randn(len(cx))
    cy = cy + stddev * np.random.randn(len(cy))
    centers = np.array([np.array((x, y)) for x in cx for y in cy])
    return (centers, rad)


def generate_place_fields_CI(n_fields, rad_range, exclusion_param):
    radii = (rad_range[1] - rad_range[0]) * np.random.random_sample(
        n_fields
    ) + rad_range[0]
    field_c = []
    for field in range(n_fields):
        # print('field = ', field)
        # pick a center in range -1, 1
        c = 2 * np.random.rand(2) - 1
        if field == 0:
            field_c.append(c)
            continue
        added = False
        collision = False
        trie = 0
        maxtries = 100
        # print('field c', field_c)
        while trie < maxtries and added == False:
            # print('trie', trie)
            for cbar_ind, cbar in enumerate(field_c):
                # print('cbar', cbar)
                if np.linalg.norm(c - cbar) < exclusion_param * radii[cbar_ind]:
                    # already a field there, try again
                    # print('collision')
                    c = 2 * np.random.rand(2) - 1
                    collision = True
                    break
            if collision:
                trie += 1
                collision = False
                continue
            else:
                field_c.append(c)
                added = True
        if not added:
            field_c.append(c)
    return (np.array(field_c), radii)


def generate_spikes_gaussian(paths, fields, max_rate, sigma):

    ncell, dim = fields.shape
    ntrial, nwin, _ = paths.shape

    spikes = np.zeros((ncell, nwin, ntrial))

    P1 = paths[:, :, np.newaxis, :]
    C1 = fields[np.newaxis, np.newaxis, :, :]

    P1 = np.tile(P1, [1, 1, ncell, 1])
    C1 = np.tile(C1, [ntrial, nwin, 1, 1])

    S = P1 - C1
    M = np.einsum("ijkl, ijkl->ijk", S, S)
    probs = max_rate * np.exp(-1 * M / (2 * sigma ** 2))
    spikes = 1 * np.greater(probs, np.random.random(np.shape(probs)))
    return np.einsum("ijk->kji", spikes)


def generate_spikes(paths, fields, max_rate, rads):

    ncell, dim = fields.shape
    ntrial, nwin, _ = paths.shape

    spikes = np.zeros((ncell, nwin, ntrial))

    P1 = paths[:, :, np.newaxis, :]
    C1 = fields[np.newaxis, np.newaxis, :, :]

    P1 = np.tile(P1, [1, 1, ncell, 1])
    C1 = np.tile(C1, [ntrial, nwin, 1, 1])

    S = P1 - C1
    M = np.einsum("ijkl, ijkl->ijk", S, S)
    # SIGMA = sigma*np.ones(M.shape)
    # forgot square root
    M = np.sqrt(M)
    SIGMA = np.tile(rads[np.newaxis, np.newaxis, :], (ntrial, nwin, 1))
    # if distance is less than sigma, then p = max_rate
    probs = max_rate * np.less(M, SIGMA)
    spikes = 1 * np.greater(probs, np.random.random(np.shape(probs)))
    return np.einsum("ijk->kji", spikes)


def spikes_to_dataframe(spikes, fs, nsecs):
    (ncells, nwin, ntrial) = spikes.shape
    spikes_frame = pd.DataFrame(columns=["cluster", "time_samples", "recording"])
    trials_frame = pd.DataFrame(columns=["stimulus", "time_samples", "stimulus_end"])
    for trial in range(ntrial):
        for cell in range(ncells):
            cellspikes = (
                np.nonzero(spikes[cell, :, trial])[0] + trial * (nsecs + 2) * fs
            )
            celldict = {
                "cluster": len(cellspikes) * [cell],
                "time_samples": cellspikes,
                "recording": len(cellspikes) * [0],
            }
            cellframe = pd.DataFrame(celldict)
            spikes_frame = spikes_frame.append(cellframe, ignore_index=True)
        trial_frame = pd.DataFrame(
            {
                "stimulus": "joe",
                "time_samples": trial * (nsecs + 2) * fs,
                "stimulus_end": (trial * (nsecs + 2) + nsecs) * fs,
                "recording": 0,
            },
            index=[0],
        )
        trials_frame = trials_frame.append(trial_frame, ignore_index=True)
    clusters_frame = pd.DataFrame(
        {"cluster": range(ncells), "quality": ncells * ["Good"]}
    )
    return (spikes_frame.sort_values(by="time_samples"), trials_frame, clusters_frame)


def plot_environment(env, fields, sigma):
    # Plot environments
    plt.style.use("/home/brad/code/NeuralTDA/gentnerlab.mplstyle")

    rad = sigma

    fig = plt.figure()
    # plt.plot(pths1[0, :, 0], pths1[0, :, 1], alpha=0.5)
    ax = fig.add_subplot(111)
    # ax = plt.gca()
    for hole in env.holes:

        h1 = plt.Circle(hole, env.hole_rad, fill=False, color="r")
        ax.add_artist(h1)
    for field in fields:
        h2 = plt.Circle(field, rad, fill=True, color="g", alpha=0.15)
        ax.add_artist(h2)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.title('Environment 1')
    ax.set_aspect("equal")


class EnvironmentSimulation:
    def __init__(
        self,
        L,
        v,
        hole_radius,
        nseconds,
        fs,
        ncells,
        ntrials,
        max_rate_hz,
        sigma,
        max_hole,
        nrepeats,
        exclusion_param,
    ):

        self.L = L
        self.v = v
        self.vel = v * L

        self.hole_radius = hole_radius
        self.nseconds = nseconds
        self.fs = fs
        self.nwin = nseconds * fs

        self.ncells = ncells
        self.ntrials = ntrials
        self.max_rate_hz = max_rate_hz
        self.max_rate = max_rate_hz / fs
        self.sigma = sigma
        self.sigmaL = sigma * L

        self.max_hole = max_hole
        self.nrepeats = nrepeats
        self.exclusion_param = exclusion_param
        self.num_envs = max_hole * nrepeats

        self._config()

    def _config(self):

        print("Generating environments...")
        self.envs = generate_environments(
            self.max_hole, self.hole_radius, self.nrepeats
        )

        print("Generating Place Fields...")
        self.fields, self.rads = generate_place_fields_CI(
            self.ncells, [self.sigmaL, self.sigmaL], self.exclusion_param
        )

        print("Generating spikes...")
        self.spikes = []
        self.paths = []
        for env1 in self.envs:
            pths1 = generate_paths(env1, self.nwin, self.ntrials, self.vel)
            self.paths.append(pths1)
            spikes1 = generate_spikes(pths1, self.fields, self.max_rate, self.rads)
            self.spikes.append(spikes1)

    def reconfigure(self):

        self._config()

    def compute_simplicial_complexes(self, windt, dtovr, thresh):

        self.graphs = []
        for ind, spikes1 in enumerate(self.spikes):
            (f, tmpf) = tempfile.mkstemp()
            os.close(f)
            tspikes, ttrials, tclust = spikes_to_dataframe(
                spikes1, fs=self.fs, nsecs=self.nseconds
            )
            print("Binning data...")
            tp2.build_binned_file_quick(
                tspikes,
                ttrials,
                tclust,
                windt,
                self.fs,
                ["Good"],
                [0, 0],
                tmpf,
                dt_overlap=dtovr,
            )

            print("Computing simplicial complexes...")
            with h5.File(tmpf, "r") as bf:
                poptens = np.array(bf["joe"]["pop_tens"])
                ncell, nwin, ntrial = poptens.shape
                for trial in range(ntrial):
                    binmat = ss.binnedtobinary(poptens[:, :, trial], thresh)
                    maxsimps = ss.binarytomaxsimplex(binmat, rDup=False)
                    g = ss.stimspacegraph_nx(maxsimps, self.ncells, stimuli=None)
                    self.graphs.append((g, maxsimps, binmat))

            os.remove(tmpf)

    def mds_embed(self, env_num):

        g = self.graphs[env_num][0]
        maxsimps = self.graphs[env_num][1]
        binmat = self.graphs[env_num][2]
        pths1 = self.paths[env_num]
        stim = pths1[0, :, :].T
        self.embed_pts, self.dmat, self.sorted_node_list = ss.mds_embed(g)
        self.x, self.y = ss.prepare_affine_data(
            binmat, stim, self.embed_pts, self.sorted_node_list
        )

        self.L = lambda a: ss.affine_loss(a, self.x, self.y, 2, 2)

    def fit_affine(self):

        a_min = fmin(self.L, [1, 0, 0, 1, 0, 0], maxfun=10000)
        self.y_embed = ss.affine_transform(a_min, self.x, 2, 2)
