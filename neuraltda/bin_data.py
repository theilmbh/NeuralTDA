#!/usr/bin/env python
import sys
import os
import argparse
import topology


def get_args():

	parser = argparse.ArgumentParser(description='Bin Spiking Data')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('bin_def_file', type=str, help='location of the file describing the binning parameters')
	parser.add_argument('bin_id', type=str, help='ID for binning')
	parser.add_argument('nshuffs', type=int, default=0, help='Number of shuffles.  0 if no controls to be made')

	return parser.parse_args()

def main():

	args = get_args()
	block_path = os.path.abspath(args.block_path)
	bin_def_file = os.path.abspath(args.bin_def_file)
	bin_id = args.bin_id
	nshuffs = args.nshuffs
	topology.prep_and_bin_data(block_path, bin_def_file, bin_id, nshuffs)


if __name__ == '__main__':
	main()