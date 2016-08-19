#!/usr/bin/env python
import sys
import os
import argparse
import topology
import glob
import logging
import datetime

def get_args():

	parser = argparse.ArgumentParser(description='Calculate CI topology on permuted binned dataset')
	parser.add_argument('analysis_id', type=str, help='A unique ID string for this run')
	parser.add_argument('threshold', type=float, help= 'Threshold for Cell Groups')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('binned_path', type=str, help='Path to folder containing binned')

	return parser.parse_args()

def main():

	args = get_args()

	block_path = os.path.abspath(args.block_path)
	analysis_id = args.analysis_id
	binned_data_files = glob.glob(os.path.join(args.binned_path, '*.binned'))
	
	# Logging facilities
	# Make logging dir if doesn't exist
	logging_dir = os.path.join(os.getcwd(), 'logs/')
	if not os.path.exists(logging_dir):
		os.makedirs(logging_dir)
	logging_filename = 'calculate_CI_topology_recursive-' +  datetime.datetime.now().strftime('%d%m%y%H%M%S') + '.log'
	logging_file = os.path.join(logging_dir, logging_filename)
	logging.basicConfig(filename=logging_file, level=logging.DEBUG, format='%(asctime)s %(message)s')
	logging.info('Starting calculate_CI_topology_recursive.')

	for bdf in binned_data_files:
		topology.calc_CI_bettis_hierarchical_binned_data(analysis_id, bdf, block_path, args.threshold)

if __name__ == '__main__':
	main()