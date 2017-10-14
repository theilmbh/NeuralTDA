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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static unsigned int v1[MAXDIM] = {1,2,3,4,0,0,0,0,0,0};
static unsigned int v2[MAXDIM] = {1,2,4,8,0,0,0,0,0,0};
static unsigned int v3[MAXDIM] = {1,2,4,8,6,10,0,0,0,0};

int test_compute_chain_groups()
{
    struct Simplex *max_simps[3];

    struct Simplex * s1 = create_empty_simplex();
    add_vertex(s1, 3);
    add_vertex(s1, 10);
    add_vertex(s1, 1);
    add_vertex(s1, 7);

    struct Simplex * s2 = create_empty_simplex();
    add_vertex(s2, 3);
    add_vertex(s2, 5);
    add_vertex(s2, 6);
    add_vertex(s2, 7);

    struct Simplex * s3 = create_empty_simplex();
    for (int i=10; i>=0; i--) {
        add_vertex(s3, i);
    }
    
    max_simps[0] = s1;
    max_simps[1] = s2;
    max_simps[2] = s3;
    SCG * scg1 = get_empty_SCG();

    clock_t start, end;
    double cpu_time;
    start = clock();
    compute_chain_groups(max_simps, 3, scg1);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    print_SCG(scg1);
    printf("CPU Time: %f milliseconds\n", cpu_time*1000.);
    return 1;
}

int test_scg_list_union()
{
    /* Test union of two scg lists */
    int retcode = 1;
    int dim = 3;

    SCG * scg1 = get_empty_SCG();
    SCG * scg2 = get_empty_SCG();
    
    struct Simplex * s1 = create_simplex(v1, dim);
    struct Simplex * s2 = create_simplex(v2, dim);
    struct Simplex * s3 = create_empty_simplex();
    for (int i=90; i>=1; i--) {
        add_vertex(s3, i);
    }
    
    scg_add_simplex(scg1, s1);
    scg_add_simplex(scg2, s2);
    scg_add_simplex(scg1, s3);

    printf("SCG1\n");
    print_SCG(scg1);
    printf("SCG2\n");
    print_SCG(scg2);

    scg_list_union(scg1, scg2);
    if ((!simplex_list_isin(scg2->x[3], s1))) {
        retcode = 0;
    }
    print_SCG(scg2);
    scg_list_union(scg1, scg2);
    print_SCG(scg2);
    return retcode;
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
    int dim = 3;

    struct simplex_list *new_sl = get_empty_simplex_list();
    struct Simplex * s1 = create_simplex(v1, dim);
    struct Simplex * s2 = create_simplex(v2, dim);

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
    struct Simplex *s1 = malloc(sizeof(struct Simplex));
    struct Simplex *s2 = malloc(sizeof(struct Simplex));
    struct Simplex *s3 = malloc(sizeof(struct Simplex));

    /* Fill in simplices */
    s1->dim = 3;
    s2->dim = 3;
    s3->dim = 3;

    memcpy(s1->vertices, v1, MAXDIM*sizeof(unsigned int));
    memcpy(s2->vertices, v1, MAXDIM*sizeof(unsigned int));
    memcpy(s3->vertices, v2, MAXDIM*sizeof(unsigned int));

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
    printf("Tests succeeded \n");
    return 0;
}
