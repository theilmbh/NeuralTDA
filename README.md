[CI]: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000205
# NeuralTDA: Topological Data Analysis for Extracellular Neural Recordings

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

### Directory Structure

The top level directory for all computations is the block_path.  This is the directory that contains the .kwik file an all auxiliary files/folders that result from spike sorting.  There is one block_path for each recording block (typically this means that each site from each penetration for each bird will have its own block).

This code produces three main subdirectories: "binned_data", "topology", and "figures".  

The "binned_data" subdirectory contains all of the binned data files in hdf5 format.  The subdirectories of "block_path/binned_data/" are labeled by the parameter "bin_id".  In principle this id can be any string, but standard practice is to label by date and time of the initiation of the binning procedure in the format DDMMYYHHMM.

Within a bin_id subfolder, there appear files ending in .binned.  The names of these files come from the binning definitions file passed to the binning procedure.  They correspond to the binning parameters defined in the definitions file.  See "standard_good_binnings.txt" for an example. 

The permutation and shuffling procedures create subdirectories of whichever directory they are directed to operate on.  Permutation creates the subfolder "permuted_binned" and shuffling creates the subfolder "shuffled_controls".  

The subdirectories of "block_path/topology/" are also named by an analysis_id.  Standard procedure is to follow the same convention for assigning analysis_ids as assigning bin_ids.  Lately, the procedure also appends a "_real" or "_shuffled" suffix to the analysis ID in order to distinguish which directories contain topology computations from the same datasets in raw form or in shuffled form.  This labelling is necessary for the plotting routines to find which curves to plot as raw data or shuffled control data.  

The subdirectories of each analysis ID correspond to the names of the binned data files that the topology routines were instructed to compute from.  Each of these subdirectories contains all of the temporary files for computation of topology as well as .pkl files corresponding to the betti curves for each stimulus. 


