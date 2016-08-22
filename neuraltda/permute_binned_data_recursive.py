#!/usr/bin/env python
import sys
import os
import argparse
import topology
import logging
import datetime

console_script_name = 'permute_binned_data_recursive'

def get_args():

	parser = argparse.ArgumentParser(description='Permute Binned Spiking Data')
	parser.add_argument('path_to_binned', type=str, help='Path to folder'
													 'containing the source binned data files')
	
	parser.add_argument('n_cells_in_perm', type=int, default=1, help='Number of cells in each perm')
	parser.add_argument('nperms', type=int, default=1, help='Number of permutations to create')

	return parser.parse_args()

def setup_logging(func_name):

	# Logging facilities
	# Make logging dir if doesn't exist
	logging_dir = os.path.join(os.getcwd(), 'logs/')
	if not os.path.exists(logging_dir):
		os.makedirs(logging_dir)
	logging_filename = '{}-'.format(func_name) + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
	logging_file = os.path.join(logging_dir, logging_filename)
	logging.basicConfig(filename=logging_file, level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
	logging.info('Starting {}.'.format(func_name))

def main():

	args = get_args()
	path_to_binned = os.path.abspath(args.path_to_binned)
	n_cells_in_perm = args.n_cells_in_perm
	nperms = args.nperms

	setup_logging(console_script_name)
	topology.make_permuted_binned_data_recursive(path_to_binned, n_cells_in_perm, nperms)


if __name__ == '__main__':
	main()