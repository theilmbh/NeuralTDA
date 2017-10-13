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


int test_simplex_equals()
{
    /*  Test simplex comparison */
    struct Simplex *s1 = malloc(sizeof(struct Simplex));
    struct Simplex *s2 = malloc(sizeof(struct Simplex));
    struct Simplex *s3 = malloc(sizeof(struct Simplex));

    /* Fill in simplices */
    s1->dim = 3;
    s2->dim = 3;
    s3->dim = 3;

    static unsigned int v1[MAXDIM] = {1,2,3,4,0,0,0,0,0,0};
    static unsigned int v2[MAXDIM] = {1,2,4,8,0,0,0,0,0,0};
    memcpy(s1->vertices, v1, MAXDIM*sizeof(unsigned int));
    memcpy(s2->vertices, v1, MAXDIM*sizeof(unsigned int));
    memcpy(s3->vertices, v2, MAXDIM*sizeof(unsigned int));

    if ((!simplex_equals(s1, s2)) || (simplex_equals(s1, s3))) {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (!test_simplex_equals()) {
        printf("Simplex equals fails\n");
        exit(-1);
    }
    printf("Tests succeeded \n");
    return 0;
}
