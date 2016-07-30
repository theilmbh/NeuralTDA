[CI]: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000205
# NeuralTDA

## Topological Data Analysis for Extracellular Neural Recordings
## Brad Theilman

* * *

## Overview
This code is designed to work with the @Gentnerlab in-house extracellular data pipelines,
contained in the repositories [klusta_pipeline](http://github.com/gentnerlab/klusta_pipeline) and [ephys-analysis](http://github.com/gentnerlab/ephys-analysis)

This code was originally designed to perform similar analyses to those presented in the 2008 paper ["Cell groups reveal struture of stimulus space"][CI] by Curto & Itskov.

As time progresses, more topological data analyses will be added.

## Process

The analysis can be divided into stages:

Bin Data -> Compute Topological Invariants -> Make Plots

### Binning Data

The first step of the analysis is to discretize the population spiking activity.  This is done by subdividing time intervals in the recording (for example, each trial of a stimulus presentation), into windows.  For each cluster (putative neuron) in the recording, the number of spikes is computed for each window in the subdivision.  

The result is an N_clusters by N_windows matrix representing the firing rate of each cluster in that window.  These matrices are stored in an hdf5 file containing the hierarchy of stimuli and trials, along with metadata needed for further topological analysis.

#### Permutations and Shuffling

To facilitate computation of topological invariants from datasets containing many clusters, there are functions to take a previously binned dataset and select random subsets of the total number of clusters and store them in a new hdf5 file.

To produce control datasets, there are functions which, for each stimulus, for each trial, and for each cluster, shuffle the windows in time independently. This destroys spatiotemporal correlations in the population allowing a "null distribution" of topological invariants to be computed. 
