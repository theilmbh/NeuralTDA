#!/usr/bin/env python
import sys
import os
import logging
import datetime
import argparse
import build_space
import glob

cs_name = 'make_pf_graph_plots'

def get_args():

	parser = argparse.ArgumentParser(description='Permute Binned Spiking Data')
	parser.add_argument('binned_path', type=str, help='Path to folder'
													 'containing the source binned data files')

	parser.add_argument('threshold', type=float, default=4.0, help='Threhsold')
	parser.add_argument('savepath', type=str)

	return parser.parse_args()

def setup_logging(func_name):

	# Logging facilities
	# Make logging dir if doesn't exist
	logging_dir = os.path.join(os.getcwd(), 'logs/')
	if not os.path.exists(logging_dir):
		os.makedirs(logging_dir)
	logging_filename = '{}-'.format(func_name) + datetime.datetime.now().strftime('%d%m%y%H%M%S') + '.log'
	logging_file = os.path.join(logging_dir, logging_filename)
	logging.basicConfig(filename=logging_file, level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
	logging.info('Starting {}.'.format(func_name))

def main():

	args = get_args()
	binned_data_files = glob.glob(os.path.join(args.binned_path, '*.binned'))
	
	setup_logging(cs_name)
	for bdf in binned_data_files:
		build_space.make_pf_graph_plots(bdf, args.threshold, args.savepath)


if __name__ == '__main__':
	main()