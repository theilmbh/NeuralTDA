CC=gcc
CFLAGS= -std=c99 -g -Wall -I.
DEPS= simplex.h hash_table.h boundary_op.h
OBJ = simplex.o hash_table.o test_simplex.o boundary_op.c

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

test_simplex:  $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

prof: $(OBJ)
	gcc -pg -o test_simplex $^ $(CFLAGS)
