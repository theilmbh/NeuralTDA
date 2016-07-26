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
            'shuffle_data = neuraltda.make_shuffled_binned_controls:main',
            'shuffle_data_recursive = neuraltda.make_shuffled_binned_controls_hierarchical:main',
            'calc_CI_topology_permuted_binned = neuraltda.calculate_CI_topology_from_permuted_binned:main',
            'calc_CI_topology_hierarchical = neuraltda.calculate_CI_topology_hierarchical_binned:main', 
            'make_plots = neuraltda.make_plots:main'
        ],
    },
)