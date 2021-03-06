/*
 * =====================================================================================
 *
 *       Filename:  test_simplex.c
 *
 *    Description:  Unit tests for simplex.c
 *
 *        Version:  1.0
 *        Created:  10/12/2017 05:07:17 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include "simplex.h"
#include "hash_table.h"
#include "boundary_op.h"
#include "slse.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_matrix.h>

int test_boundary_op()
{
    struct Simplex *max_simps[3];

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 0);
    add_vertex(s1, 1);
    add_vertex(s1, 2);
    add_vertex(s1, 3);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 0);
    add_vertex(s2, 1);
    add_vertex(s2, 4);
    add_vertex(s2, 5);

    struct Simplex * s3 = create_empty_simplex();
    for (int i=7; i>=5; i--) {
        add_vertex(s3, i);
    }
    max_simps[0] = s1;
    max_simps[1] = s2;
    max_simps[2] = s3;
    SCG * scg1 = get_empty_SCG();
    compute_chain_groups(max_simps, 3, scg1);
    
    struct bdry_op_dict * bdry_op_1 = compute_boundary_operator(s2);
    int * out_vec = bdry_canonical_coordinates(bdry_op_1, scg1->x[1],
                                          scg1->cg_dim[1]);
    print_SCG(scg1);    
    printf("BOUNDARY OPERATOR\n");
    for (int i = 0; i < scg1->cg_dim[1]; i++) {
        printf("%d ", out_vec[i]);

    }
    printf ("\nBOUNDARY MATRIX\n");
    int srcd = scg1->cg_dim[2];
    int trgd = scg1->cg_dim[1];
    int * out_mat = compute_boundary_operator_matrix(scg1, 2);
    for (int i = 0; i < trgd; i++) {
        for (int j = 0; j < srcd; j++) {
            printf("%2d, ", out_mat[i*srcd + j]);
        }
        printf(";\n");
    }
    printf("\n");
    srcd = scg1->cg_dim[3];
    trgd = scg1->cg_dim[2];
    out_mat = compute_boundary_operator_matrix(scg1, 3);
    for (int i = 0; i < trgd; i++) {
        for (int j = 0; j < srcd; j++) {
            printf("%2d, ", out_mat[i*srcd + j]);
        }
        printf(";\n");
    }
    printf("\n\nLAPLACIAN\n");
    int * laplacian = compute_simplicial_laplacian(scg1, 2);
    for (int i = 0; i<scg1->cg_dim[2]; i++) {
        for (int j=0; j<scg1->cg_dim[2]; j++) {
            printf("%2d, ", laplacian[i*scg1->cg_dim[2] + j]);
        }
        printf(";\n");
    }
    printf("\n");
    
    gsl_matrix * L = to_gsl(laplacian, trgd);
    double div = KL_divergence(L, L, 0.15);
    printf("Div: %f\n", div);

}
int test_compute_chain_groups()
{
    struct Simplex *max_simps[4];

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 3);
    add_vertex(s1, 110);
    add_vertex(s1, 1);
    add_vertex(s1, 7);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 3);
    add_vertex(s2, 5);
    add_vertex(s2, 6);
    add_vertex(s2, 7);

    struct Simplex * s3 = create_empty_simplex();
    for (int i=7; i>=5; i--) {
        add_vertex(s3, i);
    }
    struct Simplex * s4 = create_empty_simplex();
    for (int i=3; i>=0; i--) {
        add_vertex(s4, i);
    }
    
    max_simps[0] = s1;
    max_simps[1] = s2;
    max_simps[2] = s3;
    max_simps[3] = s4;
    SCG * scg1 = get_empty_SCG();

    clock_t start, end;
    double cpu_time;
    start = clock();
    compute_chain_groups(max_simps, 4, scg1);
    end = clock();
    
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    print_SCG(scg1);
    printf("CPU Time: %f milliseconds\n", cpu_time*1000.);
    free_SCG(scg1);
    return 1;
}

int test_scg_list_union()
{
    /* Test union of two scg lists */
    int retcode = 1;

    SCG * scg1 = get_empty_SCG();
    SCG * scg2 = get_empty_SCG();
    
    //struct Simplex * s1 = create_simplex(v1, dim);
    //struct Simplex * s2 = create_simplex(v2, dim);
    struct Simplex * s3 = create_empty_simplex();
    for (int i=4; i>=1; i--) {
        add_vertex(s3, i);
    }

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 1);
    add_vertex(s1, 34444);
    add_vertex(s1, 10);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 1);
    add_vertex(s2, 4);
    add_vertex(s2, 11);
    
    scg_add_simplex(scg1, s1);
    scg_add_simplex(scg2, s2);
    scg_add_simplex(scg1, s3);

    printf("SCG1\n");
    print_SCG(scg1);
    printf("SCG2\n");
    print_SCG(scg2);

    scg_list_union_hash(scg1, scg2, NULL);
    if ((!simplex_list_isin(scg2->x[2], s1))) {
        retcode = 0;
    }
    print_SCG(scg2);
    scg_list_union_hash(scg1, scg2, NULL);
    print_SCG(scg2);
    return retcode;
}

int test_hash_table()
{
    struct simplex_hash_entry **table = get_empty_hash_table();
    struct Simplex * s3 = create_empty_simplex();
    for (int i=4; i>=1; i--) {
        add_vertex(s3, i);
    }

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 1);
    add_vertex(s1, 34444);
    add_vertex(s1, 10);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 1);
    add_vertex(s2, 4);
    add_vertex(s2, 11);

    check_hash(table, s1); /* Should put s1 in hash table */  
    check_hash(table, s2); 
    if (check_hash(table, s1)) {
        printf("In hash table\n");
        return 1;
    } /* Should already be in there */
    return 0;
}

int test_int_from_simplex()
{
    unsigned int N;
    int retcode = 1;
    
    struct Simplex * s = create_empty_simplex();
    add_vertex(s, 3);
    add_vertex(s, 5);
    add_vertex(s, 4);
    print_simplex(s);
    N = integer_from_simplex(s);
    if (N != 7) {
        retcode = 0;
    }
    return retcode;
}

int test_add_remove_simplex()
{
    int retcode = 1;

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 1);
    add_vertex(s1, 34444);
    add_vertex(s1, 10);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 1);
    add_vertex(s2, 4);
    add_vertex(s2, 11);
    struct simplex_list *new_sl = get_empty_simplex_list();

    add_simplex(new_sl, s1);
    if ((!simplex_list_isin(new_sl, s1)) || (simplex_list_isin(new_sl, s2))) {
        retcode = 0;
        return retcode;
    }

    add_simplex(new_sl, s2);
    if ((!simplex_list_isin(new_sl, s1)) || (!simplex_list_isin(new_sl, s2))) {
        retcode = 0;
        return retcode;
    }

    new_sl = remove_simplex(new_sl, s2);
    if ((!simplex_list_isin(new_sl, s1)) || (simplex_list_isin(new_sl, s2))) {
        retcode = 0;
        return retcode;
    }
    return retcode;
}

int test_simplex_equals()
{
    int retcode = 0;
    /*  Test simplex comparison */
    struct Simplex * s3 = create_empty_simplex();
    for (int i=4; i>=1; i--) {
        add_vertex(s3, i);
    }

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 1);
    add_vertex(s1, 4);
    add_vertex(s1, 11);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 1);
    add_vertex(s2, 4);
    add_vertex(s2, 11);

    if ((simplex_equals(s1, s2)) && (!simplex_equals(s1, s3))) {
        retcode = 1;
    }

    free(s1);
    free(s2);
    free(s3);
    return retcode;
}

int main(int argc, char **argv)
{
    if (!test_hash_table()) {
        printf("hash table fails\n");
        exit(-1);
    }
    if (!test_simplex_equals()) {
        printf("Simplex equals fails\n");
        exit(-1);
    }
    if (!test_add_remove_simplex()) {
        printf("Add Remove Simplex fails\n");
        exit(-1);
    }
    if (!test_scg_list_union()) {
        printf("SCG list union fails\n");
        exit(-1);
    }
    if (!test_int_from_simplex()) {
        printf("int from simplex fails\n");
        exit(-1);
    }
    if (!test_compute_chain_groups()) {
        printf("compute chain groups fails\n");
        exit(-1);
    }
    test_boundary_op();
    printf("Tests succeeded \n");
    printf("Ncollisions: %d\n", ncollisions);
    return 0;
}
