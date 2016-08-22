#!/usr/bin/env python
import sys
import os
import argparse
import topology
import logging
import datetime

cs_name = 'bin_data' 

def get_args():

	parser = argparse.ArgumentParser(description='Bin Spiking Data')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('bin_def_file', type=str, help='location of the file describing the binning parameters')
	parser.add_argument('bin_id', type=str, help='ID for binning')
	parser.add_argument('nshuffs', type=int, default=0, help='Number of shuffles.  0 if no controls to be made')

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
	block_path = os.path.abspath(args.block_path)
	bin_def_file = os.path.abspath(args.bin_def_file)
	bin_id = args.bin_id
	nshuffs = args.nshuffs

	setup_logging(cs_name)
	topology.prep_and_bin_data(block_path, bin_def_file, bin_id, nshuffs)


if __name__ == '__main__':
	main()