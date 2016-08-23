#!/usr/bin/env python
import sys
import os
import argparse
import topology
import logging
import datetime

cs_name = 'make_shuffled_controls_recursive'

def get_args():

	parser = argparse.ArgumentParser(description='Permute Binned Spiking Data')
	parser.add_argument('path_to_binned', type=str, help='Path to folder'
													 'containing the source binned data files')

	parser.add_argument('nshuffs', type=int, default=1, help='Number of shuffles to create')

	return parser.parse_args()

def main():

	args = get_args()
	path_to_binned = os.path.abspath(args.path_to_binned)
	nshuffs = args.nshuffs
	topology.setup_logging(cs_name)
	topology.make_shuffled_controls_recursive(path_to_binned, nshuffs)



if __name__ == '__main__':
	main()