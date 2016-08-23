#!/usr/bin/env python
import sys
import os
import logging
import datetime
import argparse
import build_space
import topology
import glob

cs_name = 'make_pf_graph_plots'

def get_args():

	parser = argparse.ArgumentParser(description='Permute Binned Spiking Data')
	parser.add_argument('binned_path', type=str, help='Path to folder'
													 'containing the source binned data files')

	parser.add_argument('threshold', type=float, default=4.0, help='Threhsold')
	parser.add_argument('savepath', type=str)

	return parser.parse_args()

def main():

	args = get_args()
	binned_data_files = glob.glob(os.path.join(args.binned_path, '*.binned'))
	
	topology.setup_logging(cs_name)
	for bdf in binned_data_files:
		build_space.make_pf_graph_plots(bdf, args.threshold, args.savepath)


if __name__ == '__main__':
	main()