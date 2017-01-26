import neuraltda.topology as tp

blockPath='./'
winSize = 25.0
segmentInfo = {'period':1}
ncellsperm = 30
nperms = 20
nshuff = 1
thresh=4.0

tp.setup_logging('B1083_P03S05_25ms_4x')
bfdict = tp.dag_bin(blockPath, winSize, segmentInfo, ncellsperm, nperms, nshuff)
tp.dag_topology(blockPath, thresh, bfdict)


