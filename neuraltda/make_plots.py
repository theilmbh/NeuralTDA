#!/usr/bin/env python
import logging
import datetime
import sys
import os
import argparse
import topology_plots

cs_name = 'make_plots'

def get_args():

	parser = argparse.ArgumentParser(description='Make all plots from previously computed topology')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing the data')
	
	parser.add_argument('analysis_id', type=str, help='Number of cells in each perm')
	parser.add_argument('maxbetti', type=int, default=3, help='Number of permutations to create')
	parser.add_argument('maxt', type=int, default=7, help='Maximum Time Point')
	parser.add_argument('figx', type=int, default=22, help='size in inches plot x')
	parser.add_argument('figy', type=int, default=22, help='size in inches plot x')
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
	block_path = os.path.abspath(args.block_path)
	analysis_id = args.analysis_id
	maxbetti = args.maxbetti
	maxt = args.maxt
	figx = args.figx
	figy = args.figy
	figsize = (figx, figy)

	setup_logging(cs_name)
	topology_plots.make_all_plots(block_path, analysis_id, maxbetti, maxt, figsize)

if __name__ == '__main__':
	main()