#!/usr/bin/env python
import sys
import os
import argparse
import topology


def get_args():

	parser = argparse.ArgumentParser(description='Compute Cij matrix for each trial/permutation/shuffle and store in binned file')
	parser.add_argument('binned_path', type=str, help='Path to folder'
													 'containing the binned data')
	parser.add_argument('tmax', type=float, default=1, help='Maximum Time for correlations')
	return parser.parse_args()

def main():

	args = get_args()
	block_path = os.path.abspath(args.block_path)
	tmax = args.tmax
	topology.make_cij(binned_path, tmax)

if __name__ == '__main__':
	main()