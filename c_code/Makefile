CC=gcc
CFLAGS= -std=c99 -g -Wall -I.
DEPS= simplex.h hash_table.h
OBJ = simplex.o hash_table.o test_simplex.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

test_simplex:  $(OBJ)
	gcc -o $@ $^ $(CFLAGS)
