import neuraltda.topology2 as tp2
from joblib import Parallel, delayed
winSizes = [5.0, 10.0, 25.0, 50.0]
segmentInfo = [0, 0]
povers = 0.5

blockPath='./'

Parallel(n_jobs=5)(delayed(tp2.dag_bin)(blockPath, winSize, segmentInfo, dtOverlap=winSize*povers, cluster_group=['Good']) for winSize in winSizes)
