CC=gcc
CFLAGS= -fPIC -std=c99 -g -Wall -I. -I/usr/local/include -lc -lgsl -lgslcblas -lm
DEPS= simplex.h hash_table.h boundary_op.h
OBJ = simplex.o hash_table.o boundary_op.c slse.o
TARGET = libslsa.so

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET):  $(OBJ)
	gcc -o $@ $^ $(CFLAGS) -shared

