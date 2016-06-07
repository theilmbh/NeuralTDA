#!/usr/bin/env python
import sys
import os
import argparse
import topology


def get_args():

	parser = argparse.ArgumentParser(description='Calculate full-segment' 
												 'topology of an ' 
												 'extracellular dataset')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('bin_def_file', type=str, help='location of the file describing the binning parameters')
	parser.add_argument('bin_id', type=str, help='ID for binning')

	return parser.parse_args()

def main():

	args = get_args()
	block_path = os.path.abspath(args.block_path)
	bin_def_file = os.path.abspath(args.bin_def_file)
	bin_id = args.bin_id
	topology.prep_and_bin_data(block_path, bin_def_file, bin_id)

if __name__ == '__main__':
	main()