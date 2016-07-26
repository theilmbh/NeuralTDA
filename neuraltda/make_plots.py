#!/usr/bin/env python
import sys
import os
import argparse
import topology_plots


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

def main():

	args = get_args()
	block_path = os.path.abspath(args.block_path)
	analysis_id = args.analysis_id
	maxbetti = args.maxbetti
	maxt = args.maxt
	figx = args.figx
	figy = args.figy
	figsize = (figx, figy)
	topology_plots.make_all_plots(block_path, analysis_id, maxbetti, maxt, figsize)

if __name__ == '__main__':
	main()