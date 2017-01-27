import neuraltda.topology2 as tp2

blockPath='./'
winSize = 25.0
segmentInfo = {'period':1}
ncellsperm = 30
nperms = 20
nshuff = 1
thresh=4.0

threshs = [3.0, 4.0, 6.0]
winSizes = [5.0, 10.0, 25.0, 50.0]
povers = [0.0, 0.5]

birdid = 1083
site = 'P01S03'

for thresh in threshs:
	for winSize in winSizes:
		for pover in povers:
			tover = winSize*pover
			print((thresh, winSize, tover))
			tp2.setup_logging('B%d_%s_%fms_%fx_%fover', (birdid, site, winSize, thresh, tover)
			bfdict = tp2.dag_bin(blockPath, winSize, segmentInfo, ncellsperm, nperms, nshuff, dtOverlap=tover)
			aid = bfdict['analysis_id']
    		bpt = os.path.join(block_path, 'topology/')
    		analysis_dict = dict()
			rawFolder = bfdict['raw']
        	tpid_raw = aid +'-{}-raw'.format(thresh)
        	tpf_raw = os.path.join(bpt, tpid_raw)
        	rawDataFiles = glob.glob(os.path.join(rawFolder, '*.binned'))
        	for rdf in rawDataFiles:
            	TOPOLOGY_LOG.info('Computing topology for: %s' % rdf)
            	resF = tp2.computeTotalTopology(tpid_raw, rdf, block_path, thresh)
        	with open(resF, 'r') as f:
            	res = pickle.load(f)
            	analysis_dict['raw'] = res

            master_fname = aid+'-{}-masterResults.pkl'.format(thresh)
    		master_f = os.path.join(block_path, master_fname)
    		with open(master_f, 'w') as f:
            	pickle.dump(analysis_dict, f)
