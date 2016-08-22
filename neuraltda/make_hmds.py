#!/usr/bin/env python
import sys
import os
import logging
import datetime
import argparse
import build_space
import glob

cs_name = 'make_hmds'

def get_args():

	parser = argparse.ArgumentParser(description='make hyperbolic embeddings')
	parser.add_argument('hmds_path',type=str)
	parser.add_argument('binned_path', type=str, help='Path to folder'
													 'containing the source binned data files')

	parser.add_argument('threshold', type=float, default=4.0, help='Threhsold')
	parser.add_argument('savepath', type=str)
	parser.add_argument('c', type=float)
	parser.add_argument('tau', type=float)
	parser.add_argument('eps',type=float)
	parser.add_argument('eta',type=float)
	parser.add_argument('maxiter',type=int)
	parser.add_argument('maxtrial',type=int)

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
	binned_data_files = glob.glob(os.path.join(args.binned_path, '*.binned'))
	hmds_params = {'hmds_path':args.hmds_path, 'eps':args.eps, 'eta': args.eta, 'maxiter':args.maxiter, 'maxtrial': args.maxtrial, 'verbose': 1}
	dfunc_params = {'c':args.c, 'tau':args.tau}

	setup_logging(cs_name)
	for bdf in binned_data_files:
		build_space.make_pf_graph_plots(bdf, args.threshold, args.savepath, dfunc_params, hmds_params)


if __name__ == '__main__':
	main()