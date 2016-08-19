#!/usr/bin/env python
import sys
import os
import argparse
import build_space
import glob


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

def main():

	args = get_args()
	binned_data_files = glob.glob(os.path.join(args.binned_path, '*.binned'))
	hmds_params = {'hmds_path':args.hmds_path, 'eps':args.eps, 'eta': args.eta, 'maxiter':args.maxiter, 'maxtrial': args.maxtrial, 'verbose': 1}
	dfunc_params = {'c':args.c, 'tau':args.tau}
	for bdf in binned_data_files:
		build_space.make_pf_graph_plots(bdf, args.threshold, args.savepath, dfunc_params, hmds_params)


if __name__ == '__main__':
	main()