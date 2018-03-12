/*
 * =====================================================================================
 *
 *       Filename:  simplex.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/12/2017 04:52:54 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef SIMPLEX_H
#define SIMPLEX_H

#ifndef NOPYTHON
#include <Python.h>
#endif

#include "hash_table.h"

#define MAXNAME 128
#define MAXMS 12
#define MAXDIM 40 

struct Simplex {
    int vertices[MAXDIM];
    int dim;
};

struct simplex_list {
    struct Simplex * s;
    struct simplex_list * next;
    struct simplex_list * prev;
};

/* Array of simplex lists forms simplicial complex generators */
typedef struct SCG {
    struct simplex_list *x[MAXDIM];
    int cg_dim[MAXDIM]; /* Dimensions of the chain groups */
}SCG;

unsigned int num_ones(unsigned int N);
int check_bit(unsigned int N, unsigned int i);
unsigned int integer_from_simplex(struct Simplex * simp);
void get_faces_common(unsigned int N, int *verts, int dim, SCG * scg_temp);
SCG * get_faces(struct Simplex * simp);
int int_cmp(const void * a, const void * b);
void compute_chain_groups(struct Simplex ** max_simps,
                          int n_max_simps, SCG * scg_out);

/* Simplex Functions */
struct Simplex * create_simplex(unsigned int *vertices, int dim);
void free_simplex(struct Simplex * s);
int simplex_equals(struct Simplex * s1, struct Simplex * s2);
struct Simplex *get_simplex_from_integer(unsigned int N);
struct Simplex * create_empty_simplex(void);
void add_vertex(struct Simplex * s, int v);

/* Simplex List functions */
void simplex_list_free(struct simplex_list * sl);
struct simplex_list * get_empty_simplex_list(void);
void add_simplex(struct simplex_list *slist, struct Simplex *s);
void add_simplex_nocheck(struct simplex_list *slist, struct Simplex *s);
struct simplex_list * remove_simplex(struct simplex_list *slist,
                                     struct Simplex *s);
struct simplex_list * simplex_list_isin(struct simplex_list *slist,
                                        struct Simplex *);

/* SCG functions */
SCG * get_empty_SCG(void); 
void free_SCG(SCG * scg);
void scg_list_union(SCG * scg1, SCG * scg2);
void scg_list_union_hash(SCG * scg1, SCG * scg2,
                         struct simplex_hash_table *table);
void scg_add_simplex(SCG * scg, struct Simplex * s);
void scg_add_simplex_nocheck(SCG * scg, struct Simplex * s);

void scg_add_max_simplex(SCG * scg, struct Simplex * max_s);

/* print functions */
void print_simplex(struct Simplex * s);
void print_simplex_list(struct simplex_list *sl);
void print_SCG(SCG * scg);

extern int ncollisions; 

#endif

