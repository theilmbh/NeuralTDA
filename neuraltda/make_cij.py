#!/usr/bin/env python
import sys
import os
import logging
import datetime
import argparse
import topology

cs_name = 'make_cij'

def get_args():

	parser = argparse.ArgumentParser(description='Compute Cij matrix for each trial/permutation/shuffle and store in binned file')
	parser.add_argument('binned_path', type=str, help='Path to folder'
													 'containing the binned data')
	parser.add_argument('tmax', type=float, default=1, help='Maximum Time for correlations')
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
	binned_path = os.path.abspath(args.binned_path)
	tmax = args.tmax
	setup_logging(cs_name)
	topology.make_Cij(binned_path, tmax)

if __name__ == '__main__':
	main()