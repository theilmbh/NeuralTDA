##################################################
### Computes Topology by collapsing all trials ###
##################################################

import neuraltda.topology2 as tp2

blockPath='./'
segmentInfo = {'period':1}
thresh=4.0

ncellsperm = -1
nperms = 0
nshuff = 0
birdid = 1083
site = 'P03S03'

winSize = 10.0
thesh = 4.0
pover = 0.5

tover = winSize*pover
print((thresh, winSize, tover))
tp2.setup_logging('B%d_%s_%fms_%fx_%fover' % (birdid, site, winSize, thresh, tover))
bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, ncellsperm, nperms, nshuff, dtOverlap=tover)
bfdict['alltrials'] = bfdict['raw']
bfdict.pop('raw', None)
tp2.dag_topology(blockPath, thresh, bfdict)
