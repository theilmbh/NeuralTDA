## Makes a file containing the simplicial complex generators
## from the data in the directory in which this script lives
## Bradley Theilman 2017-02-07

import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
import pickle
import glob
import os

bp1 = './'
bps = [bp1]

winSize = 10.0 #ms
segmentInfo = {'period': 1}
ncellsperm = 0
nperms = 0
nshuffs = 0
thresh = 6.0
propOverlap = 0.5
dtovr = propOverlap*winSize

for blockPath in bps:
    bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, ncellsperm, nperms, nshuffs, dtOverlap=dtovr)
    bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
    sa.computeChainGroups(blockPath, bdf, thresh)