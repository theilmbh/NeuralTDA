import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
import pickle
import glob
import os

scgpath = './scg/'

scgfiles = glob.glob(os.path.join(scgpath, '*.scg'))
for scgfile in scgfiles:
	(fname, ext) = os.path.splitext(scgfile)
	if not os.path.exists(fname + '.laplacian'):
		sa.computeSimplicialLaplacians(scgfile)
	else:
		print('Laplacian already exists')
		
