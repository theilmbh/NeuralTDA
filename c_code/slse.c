/*
 * =====================================================================================
 *
 *       Filename:  slse.c
 *
 *    Description:  Simplicial Laplacian Spectral Entropy functions
 *
 *        Version:  1.0
 *        Created:  10/12/2017 08:36:33 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include <math.h>
#include <stdio.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_eigen.h>

int check_square_matrix(gsl_matrix * a)
{
    int ret = 0;
    size_t size1 = a->size1;
    size_t size2 = a->size2;
    
    if (size1 != size2) {
        ret = -1;
    } else {
        ret = size1;
    }
    return ret;
}

/* Computes the KL divergence between two density matrices.
 * Computes eigenvalues independently, sorts them, then 
 * computes divergence */
double KL_divergence(gsl_matrix * rho, gsl_matrix * sigma)
{
    double div = 0.0;
    double rval, sval;

    int i;
    size_t n;

    /* Check if they are square matrices and report size */
    if ((n = check_square_matrix(rho)) < 0) {
        printf("Rho matrix not Square!! \n");
        return 0;
    } else if ( check_square_matrix(sigma) != n) {
        printf("Rho and Sigma dimensions do not match! \n");
        return 0;
    }

    gsl_vector * rhov = gsl_vector_alloc(n);
    gsl_vector * sigmav = gsl_vector_alloc(n);

    /* Allocate workspace for eigendecomposition */
    gsl_eigen_symm_workspace * w =  gsl_eigen_symm_alloc(n);

    /* Compute eigenvalues */ 
    gsl_eigen_symm(rho, rhov, w);
    gsl_eigen_symm(sigma, sigmav, w); 

    /* Sort eigenvalues */
    gsl_sort_vector(rhov);
    gsl_sort_vector(sigmav);

    /* Compute divergence */
    for (i = 0; i < n; i++) {
        rval = gsl_vector_get(rhov, i);
        sval = gsl_vector_get(sigmav, i);
        div += rval*(log(rval) - log(sval))/M_LOG2E;
    }
    return div;
}
