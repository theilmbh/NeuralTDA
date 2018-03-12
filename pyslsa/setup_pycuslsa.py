from distutils.core import setup, Extension

pyslsa_module = Extension('pycuslsa',
						  include_dirs = ['/home/brad/code/NeuralTDA/pyslsa/slsa/'],
						  library_dirs= ['/home/brad/code/NeuralTDA/lib'],
                          libraries = ['gsl', 'gslcblas', 'm', 'slsa'],
                          sources = ['./slsa/pycuslsa.c'])
setup(name='pycuslsa', version='0.1', 
      ext_modules=[pyslsa_module])