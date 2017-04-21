import neuraltda.topology2 as tp2
from ephys import core, events, clust
import pandas as pd
from joblib import Parallel, delayed

win_size = 10.0
thresh = 13.0
segment_info = [0,0]
povers = 0.5
njobs = 10
dtovr = povers*win_size

blockPath='./'
cluster_group = ['Good', 'MUA']

widenarrow_threshold = 0.000540 # sw threshold in seconds

clusters = core.load_clusters(blockPath)
clusters = clusters[clusters.quality.isin(cluster_group)].unique()

(wide, narrow) = clust.get_wide_narrow(blockPath, clusters, widenarrow_threshold)

bfdict = tp2.dag_bin(blockPath, win_size, segment_info, cluster_group=cluster_group, dt_overlap=dtovr)
bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
sa.computeChainGroups(blockPath, bdf, thresh, clusters=wide, comment='wide')
sa.computeChainGroups(blockPath, bdf, thresh, clusters=narrow, comment='narrow')

