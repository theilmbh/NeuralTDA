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

#define MAXNAME 128
#define MAXMS 12
#define MAXDIM 10

struct Simplex {
    unsigned int vertices[MAXDIM];
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
}SCG;

unsigned int num_ones(unsigned int N);
int check_bit(unsigned int N, unsigned int i);
struct Simplex *get_simplex_from_integer(unsigned int N);
void get_faces_common(struct simplex_list face_list,
                      unsigned int N, int n_verts);
int int_cmp(const void * a, const void * b);
int simplex_equals(struct Simplex * s1, struct Simplex * s2);
void scg_list_union(SCG * scg1, SCG * scg2);
struct simplex_list * get_empty_simplex_list();
void simplex_list_free(struct simplex_list * sl);
SCG * get_empty_SCG(); 
void free_SCG(SCG * scg);
struct simplex_list * get_empty_simplex_list();
void compute_chain_groups(struct Simplex * max_simps,
                          int n_max_simps, SCG * scg_out);

#endif
