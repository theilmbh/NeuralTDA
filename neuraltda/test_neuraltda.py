#!/usr/bin/env python
import logging
import datetime
import sys
import os
import argparse
import test_pipeline
import topology as top

cs_name = 'test_neuraltda'

def get_args():

	parser = argparse.ArgumentParser(description='Make all plots from previously computed topology')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing the data')
	parser.add_argument('bin_def_file', type=str, help='Path to folder'
													 'containing the data')
	return parser.parse_args()


def main():

	args = get_args()
	block_path = os.path.abspath(args.block_path)
	analysis_id = args.analysis_id
	maxbetti = args.maxbetti
	maxt = args.maxt
	figx = args.figx
	figy = args.figy
	figsize = (figx, figy)

	time_str = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
	analysis_id = 'test_neuraltda-analysis-' + time_str
	bin_id = 'test_neuraltda-bin-' + time_str
	top.setup_logging(cs_name)
	test_pipeline.test_pipeline(block_path, bin_id, analysis_id, bin_def_file, n_cells=60, maxt=6,
				  fs=24000.0, dthetadt=2*np.pi, kappa=2, maxfr=12, n_trials=10,
				  n_cells_in_perm=40, nperms=1, thresh=4.0)

if __name__ == '__main__':
	main()