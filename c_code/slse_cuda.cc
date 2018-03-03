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


gsl_vector * cuda_get_eigenvalues(gsl_matrix *L1, size_t n)
{

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    /* Copy Matrices to double arrays */
    double *L1mat = malloc(L1->size1*L1->size2*sizeof(double));
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            L1mat[i*L1->tda +j] = gsl_matrix_get(L1, i, j);
        }
    }

    // Allocate space for eigenvalues
    double *L1v = malloc(n*sizeof(double));


    // declare device variables
    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;

    // Create solver handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(cusolver_status = CUSOLVER_STATUS_SUCCESS);

    // Copy variables to device
    err = cudaMalloc((void**)&d_A, n*n*sizeof(double));
    assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_W, n*sizeof(double));
    assert(err == cudaSuccess);


    // Get eigenvalues for matrix 1
    cudaMemcpy(d_A, L1mat, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, n, d_A, n, d_W, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // compute spectrum
    cusolver_status = cusolverDnDsyevd(cusolverH,jobz,plo, m, d_A, lda, d_W, d_work, lwork, devInfo);
    err = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == err);



}

/* Computes the KL divergence between two density matrices.
 * Computes eigenvalues independently, sorts them, then 
 * computes divergence */
double __attribute__((optimize("O0"))) KL_divergence_cuda(gsl_matrix * L1, 
                                                     gsl_matrix * L2,
                                                     double beta)
{
    double div = 0.0;
    double rval, sval;

    int i;
    size_t n;

    /* Check if they are square matrices and report size */
    if ((n = check_square_matrix(L1)) < 0) {
        printf("Rho matrix not Square!! \n");
        return 0;
    } else if ( check_square_matrix(L2) != n) {
        printf("Rho and Sigma dimensions do not match! \n");
        return 0;
    }





    gsl_vector * L1v = gsl_vector_alloc(n);
    gsl_vector * L2v = gsl_vector_alloc(n);

    gsl_vector * rhov = gsl_vector_alloc(n);
    gsl_vector * sigmav = gsl_vector_alloc(n);

    /* Allocate workspace for eigendecomposition */
    gsl_eigen_symm_workspace * w =  gsl_eigen_symm_alloc(n);

    /* Copy the matrices */
    /* We need this because GSL destroys matrices 
     * during eigenvalue computation */
    gsl_matrix * L1copy = gsl_matrix_alloc(n, n);
    gsl_matrix * L2copy = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(L1copy, L1);
    gsl_matrix_memcpy(L2copy, L2);

    /* Compute eigenvalues */ 
    gsl_eigen_symm(L1copy, L1v, w);
    gsl_eigen_symm(L2copy, L2v, w); 
    gsl_eigen_symm_free(w);

    /* Compute density eigenvalues */
    double r1, r2;
    double tr1=0, tr2=0;
    for (i = 0; i<n; i++) {
        r1 = exp(beta*gsl_vector_get(L1v, i));
        r2 = exp(beta*gsl_vector_get(L2v, i));
        gsl_vector_set(rhov, i, r1);
        gsl_vector_set(sigmav, i, r2);
        tr1 += r1; 
        tr2 += r2;
    }

    /* Sort eigenvalues */
    gsl_sort_vector(rhov);
    gsl_sort_vector(sigmav);

    /* Compute divergence */
    for (i = 0; i < n; i++) {
        rval = gsl_vector_get(rhov, i) / tr1;
        sval = gsl_vector_get(sigmav, i) / tr2;
        div += rval*(log(rval) - log(sval))/log(2.0);
    }
    /* Free Memory */
    gsl_matrix_free(L1copy);
    gsl_matrix_free(L2copy);
    gsl_vector_free(rhov);
    gsl_vector_free(sigmav);
    gsl_vector_free(L1v);
    gsl_vector_free(L2v);       
    return div;
}
