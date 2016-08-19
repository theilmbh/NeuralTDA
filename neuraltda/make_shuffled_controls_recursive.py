#!/usr/bin/env python
import sys
import os
import argparse
import topology
import logging
import datetime

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

	# Logging facilities
	# Make logging dir if doesn't exist
	logging_dir = os.path.join(os.getcwd(), 'logs/')
	if not os.path.exists(logging_dir):
		os.mkdirs(logging_dir)
	logging_filename = 'make_shuffled_controls_recursive-' + datetime.now().strftime('%d%m%y%H%M%S') + '.log'
	logging_file = os.path.join(logging_dir, logging_filename)
	logging.basicConfig(filename=logging_file, level=logging.DEBUG, format='%(asctime)s %(message)s')
	logging.info('Starting make_shuffled_controls_recursive')

	topology.make_shuffled_controls_recursive(path_to_binned, nshuffs)



if __name__ == '__main__':
	main()