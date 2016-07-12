#!/usr/bin/env python
import sys
import os
import argparse
import topology


def get_args():

	parser = argparse.ArgumentParser(description='Permute Binned Spiking Data')
	parser.add_argument('path_to_binned', type=str, help='Path to folder'
													 'containing the source binned data files')
	
	parser.add_argument('n_cells_in_perm', type=int, default=1, help='Number of cells in each perm')
	parser.add_argument('nperms', type=int, default=1, help='Number of permutations to create')

	return parser.parse_args()

def main():

	args = get_args()
	path_to_binned = os.path.abspath(args.path_to_binned)
	n_cells_in_perm = args.n_cells_in_perm
	nperms = args.nperms
	topology.make_permuted_binned_data(path_to_binned, n_cells_in_perm, nperms)


if __name__ == '__main__':
	main()