#!/usr/bin/env python
import sys
import os
import argparse
import build_space
import glob


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
	
	for bdf in binned_data_files:
		build_space.make_pf_graph_plots(bdf, args.threshold, args.savepath)


if __name__ == '__main__':
	main()