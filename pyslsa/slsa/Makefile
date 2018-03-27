CC=gcc
CFLAGS= -O0 -DNOPYTHON -fPIC -std=c99 -g -Wall -I. -I/usr/local/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -fopenmp -lc -lgsl -lgslcblas -lm
DEPS= simplex.h hash_table.h boundary_op.h
OBJ = simplex.o hash_table.o boundary_op.o slse.o slse_cuda.o
TARGET = libslsa.so

slse_cuda.o: slse_cuda.cc $(DEPS)
	nvcc -c -Xcompiler -fPIC -I/usr/local/cuda/include -I. slse_cuda.cc

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET):  $(OBJ)
	gcc -o $@ $^ $(CFLAGS) -shared

test_simplex: test_simplex.c $(DEPS) $(OBJ)
	gcc -c -o test_simplex.o $(CFLAGS) test_simplex.c
	gcc -o test_simplex test_simplex.o simplex.o hash_table.o boundary_op.o slse.o slse_cuda.o $(CFLAGS) 

libslsa.a: $(OBJ)
	ar rcs $@ $^

install: libslsa.so
	install -d /usr/local/
	install -m 644 libslsa.so /usr/local/lib 
