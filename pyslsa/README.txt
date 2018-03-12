How to install Py(cu)SLSA

- Build the libslsa library.  cd to pyslsa/slsa. Run `make libslsa.so`
- Copy libslsa.so to NeuralTDA/lib/
- Setup the python module: run CFLAGS=-O0 python setup_pycuslsa.py install
- the CFLAGS is imperative!
