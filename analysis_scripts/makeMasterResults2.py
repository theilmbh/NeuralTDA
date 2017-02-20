import neuraltda.topology2 as tp2

blockPath='./'
winSize = 25.0
segmentInfo = {'period':1}
ncellsperm = 30
nperms = 20
nshuff = 1
thresh=4.0

tp2.setup_logging('B1083_P03S05_25ms_4x')
bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, dtOverlap=dtovr)
tp2.dag_topology(blockPath, thresh, bfdict, raw=True, shuffle=True, shuffleperm=True, nperms=nperms, ncellsperm=ncellsperm)