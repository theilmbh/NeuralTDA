#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='NeuralTDA',
    version='0.0.1',
    description='Topological Data Analysis for neural data',
    author='Brad Theilman',
    author_email='bradtheilman@gmail.com',
    packages=['neuraltda/'],         
    entry_points={
        'console_scripts': [
            'calc_CI_topology = neuraltda.calculate_topology:main',
            'bin_data = neuraltda.bin_data:main',
            'calc_CI_topology_binned = neuraltda.calculate_CI_topology_from_binned:main',
            'permute_data = neuraltda.permute_binned_data:main',
            'permute_data_recursive = neuraltda.permute_binned_data_recursive:main',
            'shuffle_data = neuraltda.make_shuffled_binned_controls:main',
            'shuffle_data_recursive = neuraltda.make_shuffled_controls_recursive:main',
            'calc_CI_topology_permuted_binned = neuraltda.calculate_CI_topology_from_permuted_binned:main',
            'calc_CI_topology_recursive = neuraltda.calculate_CI_topology_recursive:main', 
            'make_plots = neuraltda.make_plots:main',
            'make_cij = neuraltda.make_cij:main',
            'make_pf_graph_plots = neuraltda.make_pf_graph_plots:main',
            'calc_cliquetop_recursive = neuraltda.calculate_cliquetop_recursive:main'
        ],
    },
)