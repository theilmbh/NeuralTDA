#!/usr/bin/env python
import sys
import os
import argparse
import topology
import glob

def get_args():

	parser = argparse.ArgumentParser(description='Calculate full-segment' 
												 'topology of an ' 
												 'extracellular dataset')
	parser.add_argument('analysis_id', type=str, help='A unique ID string for this run')
	parser.add_argument('threshold', type=float, help= 'Threshold for Cell Groups')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')
	parser.add_argument('binned_path', type=str, help='Path to folder containing binned')

	return parser.parse_args()

def main():

	args = get_args()

	block_path = os.path.abspath(args.block_path)
	analysis_id = args.analysis_id
	binned_data_files = glob.glob(binned_path)
	
	for bdf in binned_data_files:
		topology.calc_CI_bettis_binned_data(analysis_id, bdf, block_path, args.threshold)

if __name__ == '__main__':
	main()