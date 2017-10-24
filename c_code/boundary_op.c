/*
 * ============================================================================
 *
 *       Filename:  boundary_op.c
 *
 *    Description:  Routines for computing boundary operators and laplacians
 *
 *        Version:  1.0
 *        Created:  10/16/2017 12:27:36 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "boundary_op.h"
#include "simplex.h"
#include "hash_table.h"

struct bdry_op_dict * get_empty_bdry_op_dict()
{
    /* We need to make sure that the hash is clear */
    struct bdry_op_dict * out  = calloc(1, sizeof(struct bdry_op_dict));
    out->N = 0;
    return out;
}

void free_bdry_op_dict(struct bdry_op_dict * d)
{
    int i;
    for (i = 0; i < NR_BDRY_HASH; i++) {
        free_simplex(d->table[i].sp);
    }
    free(d);
}

struct bdry_op_dict * compute_boundary_operator(struct Simplex * sp)
{
    struct bdry_op_dict * out = get_empty_bdry_op_dict();
    int i, j;
    int sgn = 1;
    
    /* Null Simplex - Return the empty boundary operator */
    /* Null boundary operator - just return NULL */
    if (!sp || !out) {
        return out;
    }

    for (i = 0; i <= sp->dim; i++) {
        struct Simplex * sub = create_empty_simplex();
        for (j = 0; j <= sp->dim; j++) {
            if (j == i) continue; /* skip */
            add_vertex(sub, sp->vertices[j]);
        }
        /* We have filled in face vertices into sub */
        /* Compute sign, add to bdry table */
        add_bdry_simplex(out, sub, sgn);
        sgn *= -1;
    }

    return out;
}

void add_bdry_simplex(struct bdry_op_dict * tab, struct Simplex * sp, int sgn)
{
    /* If Table or Simplex is NULL, Do nothing */
    if ((!tab) || (!sp)) {
        printf("Null pointer encountered in add_bdry_simplex\n");
        return;
    }

    /* Check to see if simplex is already in table.. Do nothing! */
    unsigned int indx;
    if ((bdry_check_hash(tab, sp, &indx))) {
        return;
    }

    /* Else, add to table */
    tab->table[indx].sgn = sgn;
    tab->table[indx].sp = sp;
    tab->N++;
}

unsigned int bdry_check_hash(struct bdry_op_dict * tab, struct Simplex *sp,
        unsigned int *indx)
{
    /* Check hash table for simplex */
    unsigned int hash_p = simplex_hash(sp);
    unsigned int i = hash_p % NR_BDRY_HASH;

    unsigned int i2 = i;
    while ( i2 != i+1) {
        if (!tab->table[i2].sp) break;

        if (simplex_equals(tab->table[i2].sp, sp)) {
            /* Found! */
            *indx = i2; /* Set index of table entry where found */
            return 1;
        } 

        if (i2 == 0) {
            i2 = NR_BDRY_HASH - 1;
        } else {
            i2 -= 1;
        }
    }
    if (tab->N == (NR_BDRY_HASH - 1)) {
        printf("BDRYHASH table overflow\n");
    } 
    *indx = i2;
    return 0;
}

int * bdry_canonical_coordinates(struct bdry_op_dict * bdry_op,
        struct simplex_list *basis, int targ_dim)
{
    /* Create result vector */
    int * out_vec = calloc(targ_dim, sizeof(int));
    if (!out_vec) {
        printf("Unable to allocate boundary canonical coordinate vector\n");
        return out_vec;
    }

    /* Loop through simplex list, extracting sign */
    int pos = 0;
    unsigned int indx;
    while (basis) {
        if (bdry_check_hash(bdry_op, basis->s, &indx)) {
            out_vec[pos] = bdry_op->table[indx].sgn;        
        }
        basis = basis->next;
        pos++;
    }
    return out_vec;
}

gsl_matrix * compute_boundary_operator_matrix(SCG * scg, int dim)
{
    gsl_matrix *bdry_mat;
    if (dim <= 0) {
        /* Boundary operator in dimension zero is zero map */
        bdry_mat = gsl_matrix_calloc(1, 1);
        return bdry_mat;
    }

    int targ_dim = scg->cg_dim[dim-1];
    int source_dim = scg->cg_dim[dim];

    if ((targ_dim == 0) || (source_dim == 0)) {
        /* empty chain groups */
        bdry_mat = gsl_matrix_calloc(1, 1);
        return bdry_mat;
    }

    /* int *bdry_mat = calloc(targ_dim*source_dim, sizeof(int));*/
    bdry_mat = gsl_matrix_alloc(targ_dim, source_dim);

    struct simplex_list * source;
    struct simplex_list * targ;
    source = scg->x[dim];
    targ = scg->x[dim-1];

    struct bdry_op_dict *bdry_op;
    int * bdry_vec;
    
    int i=0, j=0;
    while (source) {
        bdry_op = compute_boundary_operator(source->s);
        bdry_vec = bdry_canonical_coordinates(bdry_op, targ, targ_dim);
        for (j = 0; j < targ_dim; j++) {
            gsl_matrix_set(bdry_mat, j, i, bdry_vec[j]);
            /* bdry_mat[j*source_dim + i] = bdry_vec[j];*/
        }
        i++;
        source = source->next;
        free_bdry_op_dict(bdry_op);
        free(bdry_vec);
    }
    return bdry_mat;
}

gsl_matrix * compute_simplicial_laplacian(SCG * scg, int dim)
{
    gsl_matrix * D_dim;   /* \partial_{dim} */
    gsl_matrix * D_dim_1; /* \partial_{dim+1} */
    gsl_matrix * laplacian;

    /* Check dimensions */
    int L_dim = scg->cg_dim[dim];
    int d_dim;
    if (dim > 0) {
        d_dim = scg->cg_dim[dim-1];
    } else {
        d_dim = 0;
    }
    int d_1_dim = scg->cg_dim[dim+1];

    /* Allocate result */
    if (L_dim > 0) {
        laplacian = gsl_matrix_calloc(L_dim, L_dim);
    } else {
        laplacian = gsl_matrix_calloc(1, 1);
        return laplacian;
    }

    /* Compute Boundary Operators */
    D_dim = compute_boundary_operator_matrix(scg, dim);
    D_dim_1 = compute_boundary_operator_matrix(scg, dim+1);

    /* Compute Laplacian */ 
    if (d_dim > 0) {
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0,
                       D_dim, D_dim, 0.0, laplacian);
    }
    if (d_1_dim > 0) {
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
                       D_dim_1, D_dim_1, 1.0, laplacian);
    }

    gsl_matrix_free(D_dim);
    gsl_matrix_free(D_dim_1);
    return laplacian;
}

gsl_matrix * to_gsl(int * L, size_t dim) 
{
    gsl_matrix * out;
    if (!dim) {
        out = gsl_matrix_alloc(1, 1);
        gsl_matrix_set(out, 0, 0, 0.0);
        return out;
    }
    out = gsl_matrix_alloc(dim, dim);
    size_t i, j;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            gsl_matrix_set(out, i, j, (double)L[i*dim +j]);
        }
    }
    return out;
}

void reconcile_laplacians(gsl_matrix * L1, gsl_matrix * L2,
                          gsl_matrix **L1new, gsl_matrix **L2new)
{
    gsl_matrix * temp;

    if (L1->size1 > L2->size1) {
        temp = gsl_matrix_calloc(L1->size1, L1->size2);
        for (int i = 0; i<L2->size1; i++) {
            for (int j = 0; j<L2->size2; j++) {
                gsl_matrix_set(temp, i, j, gsl_matrix_get(L2, i, j));
            }
        }
        *L1new = L1;
        *L2new = temp;
        gsl_matrix_free(L2);
        return;
    }

    if (L2->size1 > L1->size1) {
        temp = gsl_matrix_calloc(L2->size1, L2->size2);
        for (int i = 0; i<L1->size1; i++) {
            for (int j = 0; j<L1->size2; j++) {
                gsl_matrix_set(temp, i, j, gsl_matrix_get(L1, i, j));
            }
        }
        *L1new = temp;
        *L2new = L2;
        gsl_matrix_free(L1);
        return; 
    }
}
