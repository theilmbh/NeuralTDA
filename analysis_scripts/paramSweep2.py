import neuraltda.topology2 as tp2

blockPath='./'
winSize = 25.0
segmentInfo = [0,0]
clusterGroup = ['Good', 'MUA']
ncellsperm = 30
nperms = 20
nshuff = 1


threshs = [13.0, 4.0, 6.0]
winSizes = [5.0, 10.0, 25.0, 50.0]
povers = [0.5]

birdid = 1083
site = 'P01S03'

for thresh in threshs:
	for winSize in winSizes:
		for pover in povers:
			tover = winSize*pover
			print((thresh, winSize, tover))
			tp2.setup_logging('B%d_%s_%fms_%fx_%fover' % (birdid, site, winSize, thresh, tover))
			bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, dtOverlap=tover)
			tp2.dag_topology(blockPath, thresh, bfdict, raw=False, shuffle=False, shuffleperm=True, nperms=nperms, ncellsperm=ncellsperm)


