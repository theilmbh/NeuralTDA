from distutils.core import setup, Extension

pyslsa_module = Extension('pycuslsa',
						  include_dirs = ['/home/brad/code/NeuralTDA/c_code'],
						  library_dirs= ['/home/brad/code/NeuralTDA/c_code'],
                          libraries = ['gsl', 'gslcblas', 'm', 'slsa'],
                          sources = ['pycuslsa.c'])
setup(name='pycuslsa', version='0.1', 
      ext_modules=[pyslsa_module])