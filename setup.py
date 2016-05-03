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
            'bin_data = neuraltda.bin_data:main'
        ],
    },
)