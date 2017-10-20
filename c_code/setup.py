from distutils.core import setup, Extension

pyslsa_module = Extension('pyslsa',
                          libraries = ['gsl', 'gslcblas', 'm'],
                          sources = ['pyslsa.c', 'simplex.c',
                                     'hash_table.c', 'boundary_op.c'])
setup(name='pyslsa', version='0.1', 
      ext_modules=[pyslsa_module])
