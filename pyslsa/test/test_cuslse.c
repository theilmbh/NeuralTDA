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
    print_SCG(scg1);    

    size_t dim = 2;
    gsl_matrix * L = compute_simplicial_laplacian(scg1, (size_t)dim);
    int Ldim = scg1->cg_dim[dim];
    for (int i=0; i<Ldim; i++) {
        for (int j=0; j<Ldim; j++) {
            printf("%f ", gsl_matrix_get(L, i, j));
        }
        printf("\n");
    }

    double di = KL_divergence_cuda(L, L, 1.0);
    printf("Div = %f\n", di);


}

int test_cuKL()
{

    /* generate random large laplacian matrices */
    int n = 1000;
    gsl_matrix *L = gsl_matrix_alloc(n, n);
    double x = 0;

    int i, j;
    for(i=0; i<n; i++)
    {
        for(j=0; j<=i; j++)
        {
            gsl_matrix_set(L, i, j, x);
            gsl_matrix_set(L, j, i, x);
        }
    }

}

int main(int argc, char **argv)
{

    test_boundary_op();
    printf("Tests succeeded \n");
    printf("Ncollisions: %d\n", ncollisions);
    return 0;
}
