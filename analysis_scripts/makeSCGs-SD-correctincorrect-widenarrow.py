## Makes a file containing the simplicial complex generators
## from the data in the directory in which this script lives
## Bradley Theilman 2017-02-07

import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
import pickle
import glob
import os
from ephys import core, events, clust

blockPath = './'


winSize = 10.0 #ms
thresh = 13.0
povers = 0.5

cluster_group = ['Good', 'MUA']

widenarrow_threshold = 0.000540 # sw threshold in seconds

segmentInfo = [2500, 0]
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
clusters = core.load_clusters(blockPath)


clusters = clusters[clusters.quality.isin(cluster_group)]['cluster'].unique()

(wide, narrow) = clust.get_wide_narrow(blockPath, clusters, widenarrow_threshold)

correctTrials = trials[trials['correct']==True]
incorrectTrials = trials[trials['correct']==False]

bfdict = tp2.do_dag_bin_lazy(blockPath, spikes, correctTrials, clusters, fs, winSize,
                                    segmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='SD-correct')
bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf, thresh, comment='SD-correct-wide', clusters=wide)

bfdict = tp2.do_dag_bin_lazy(blockPath, spikes, correctTrials, clusters, fs, winSize,
                                    segmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='SD-correct')
bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf, thresh, comment='SD-correct-narrow', clusters=narrow)

bfdict2 = tp2.do_dag_bin_lazy(blockPath, spikes, incorrectTrials, clusters, fs, winSize,
                                    segmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='SD-incorrect')
bdf2 = glob.glob(os.path.join(bfdict2['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf2, thresh, comment='SD-incorrect-wide', clusters=wide)

bfdict2 = tp2.do_dag_bin_lazy(blockPath, spikes, incorrectTrials, clusters, fs, winSize,
                                    segmentInfo, cluster_group=['Good', 'MUA'],
                                    dt_overlap=povers*winSize, comment='SD-incorrect')
bdf2 = glob.glob(os.path.join(bfdict2['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf2, thresh, comment='SD-incorrect-narrow', clusters=narrow)