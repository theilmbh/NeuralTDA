/*
 * =====================================================================================
 *
 *       Filename:  boundary_op.h
 *
 *    Description:  Definitions for boundary operator routines
 *
 *        Version:  1.0
 *        Created:  10/16/2017 12:23:03 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef BOUNDARY_OP_H
#define BOUNDARY_OP_H

#include <gsl/gsl_matrix.h>

#include "simplex.h"
#include "hash_table.h"

#define NR_BDRY_HASH 128
struct bdry_op_vec {
    int sgn;
    struct Simplex * sp;
};

struct bdry_op_dict {
    struct bdry_op_vec table[NR_BDRY_HASH];
    int N;
    unsigned int size;
};

struct bdry_op_dict * get_empty_bdry_op_dict(void);
void free_bdry_op_dict(struct bdry_op_dict * d);
struct bdry_op_dict * compute_boundary_operator(struct Simplex * sp);
void add_bdry_simplex(struct bdry_op_dict * tab, struct Simplex * sp, int sgn);
unsigned int bdry_check_hash(struct bdry_op_dict * tab, struct Simplex *sp,
        unsigned int *indx);
int * bdry_canonical_coordinates(struct bdry_op_dict * bdry_op,
        struct simplex_list *basis, int targ_dim);
gsl_matrix * compute_boundary_operator_matrix(SCG * scg, int dim);
gsl_matrix * compute_simplicial_laplacian(SCG * scg, int dim);

void reconcile_laplacians(gsl_matrix * L1, gsl_matrix * L2,    
                          gsl_matrix **L1new, gsl_matrix **L2new);

gsl_matrix * to_gsl(int * L, size_t dim);

#endif
