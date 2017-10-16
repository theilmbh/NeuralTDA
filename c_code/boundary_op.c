/*
 * =====================================================================================
 *
 *       Filename:  boundary_op.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/16/2017 12:27:36 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>

#include "boundary_op.h"
#include "simplex.h"
#include "hash_table.h"

struct bdry_op_dict * get_empty_bdry_op_dict()
{
    struct bdry_op_dict * out  = malloc(sizeof(struct bdry_op_dict));
    out->N = 0;
    return out;
}

struct bdry_op_dict * compute_boundary_operator(struct Simplex * sp)
{
    struct bdry_op_dict * out = get_empty_bdry_op_dict();
    int i, j;
    int sgn = 1;

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
            /* found! */
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

int * compute_boundary_operator_matrix(SCG * scg, int dim)
{
    int targ_dim = scg->cg_dim[dim-1];
    int source_dim = scg->cg_dim[dim];

    int *bdry_mat = calloc(targ_dim*source_dim, sizeof(int));

    struct simplex_list * source;
    struct simplex_list * targ;
    source = scg->x[dim];
    targ = scg->x[dim-1];

    struct bdry_op_dict *bdry_op;
    int * bdry_vec;
    
    int i, j=0;
    while (source) {
        bdry_op = compute_boundary_operator(source->s);
        bdry_vec = bdry_canonical_coordinates(bdry_op, targ, targ_dim);
        for (i = 0; i < targ_dim; i++) {
            bdry_mat[j*targ_dim + i] = bdry_vec[i];
        }
        j++;
        source = source->next;
    }
    return bdry_mat;

}
