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
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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

/* Get the eigenvalues of a list of matrices using cuda streams */
extern "C" gsl_vector ** cuda_batch_get_eigenvalues(gsl_matrix * L_list[], size_t N_matrices)
{
    int i, j;
    /* Initialize cuSolver Library */
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t err;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    /* Allocate space for eigenvalues */
    size_t *sizes = (size_t *)malloc(N_matrices*sizeof(size_t));
    size_t tot_size = 0;
    for (i = 0; i < N_matrices; i++) {
        sizes[i] = L_list[i]->size1;
        tot_size += sizes[i];
    }
    double * Leigs = (double *) malloc(tot_size*N_matrices*sizeof(double));

    /* Declare Device Variables */
    double **d_As = (double **) malloc(N_matrices*sizeof(double *));
    double **d_Ws = (double **) malloc(N_matrices*sizeof(double *));
    int **devInfos = (int **) malloc(N_matrices*sizeof(int *));
    double **d_works = (double **) malloc(N_matrices*sizeof(double *));
    int *lworks = (int *) malloc(N_matrices*sizeof(int));
    
    /* Copy variables to device */
    for (i = 0; i < N_matrices; i++) {
        err = cudaMalloc((void**)&d_As[i], sizes[i]*sizes[i]*sizeof(double));
        assert(err == cudaSuccess);
        err = cudaMalloc((void**)&d_Ws[i], sizes[i]*sizeof(double));
        assert(err == cudaSuccess);
        err = cudaMalloc((void**)&devInfos[i], sizeof(int));
        assert(err == cudaSuccess);
    }

    /* Create streams */
    cudaStream_t *streams = (cudaStream_t *)malloc(N_matrices*sizeof(cudaStream_t));
    for (i = 0; i < N_matrices; i++) {
        cudaStreamCreate(&streams[i]);
    }

    /* Setup solver */
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    for (i = 0; i < N_matrices; i++) {
        cudaMemcpy(d_As[i], L_list[i]->data, sizeof(double)*sizes[i]*sizes[i], cudaMemcpyHostToDevice);
        cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, sizes[i], d_As[i], sizes[i], d_Ws[i], &lworks[i]);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        cudaMalloc(&d_works[i], sizeof(double)*lworks[i]);
    }

    /* Get eigenvalues */
    for (i = 0; i < N_matrices; i++) {
        cusolver_status = cusolverDnSetStream(cusolverH, streams[i]);
        cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, sizes[i],
                                           d_As[i], sizes[i], d_Ws[i], d_works[i],
                                           lworks[i], devInfos[i]);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    }
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    gsl_vector **ress = (gsl_vector **)malloc(N_matrices*sizeof(gsl_vector *));
    for (i = 0; i < N_matrices; i++) {
        cudaMemcpy(&Leigs[i*sizes[i]], d_Ws[i], sizes[i]*sizeof(double), cudaMemcpyDeviceToHost);
        ress[i] = gsl_vector_alloc(sizes[i]);
        for (j = 0; j < sizes[i]; j++) {
            gsl_vector_set(ress[i], j, Leigs[i*sizes[i] + j]);
        }
        cudaFree(d_As[i]);
        cudaFree(d_Ws[i]);
        cudaFree(devInfos[i]);
        cudaFree(d_works[i]);
    }
    
    free(d_As);
    free(d_Ws);
    free(devInfos);
    free(d_works);
    free(lworks);
    free(Leigs);
    cusolverDnDestroy(cusolverH);

    return ress;
}

gsl_vector * cuda_get_eigenvalues(gsl_matrix *L1, size_t n)
{

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t err;

    /* Copy Matrices to double arrays */
    int i,j;
    double *L1mat = (double *)malloc(L1->size1*L1->size2*sizeof(double));
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            L1mat[i*L1->tda +j] = gsl_matrix_get(L1, i, j);
        }
    }

    // Allocate space for eigenvalues
    double *L1v = (double *)malloc(n*sizeof(double));
    // declare device variables
    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;

    // Create solver handle
    cusolver_status = cusolverDnCreate(&cusolverH);
   //printf("cusolver_status: %d\n", cusolver_status);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Copy variables to device
    err = cudaMalloc((void**)&d_A, n*n*sizeof(double));
    assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_W, n*sizeof(double));
    assert(err == cudaSuccess);
    err = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(err == cudaSuccess);

    // Get eigenvalues for matrix 1
    cudaMemcpy(d_A, L1mat, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, n, d_A, n, d_W, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);

    // compute spectrum
    cusolver_status = cusolverDnDsyevd(cusolverH,jobz, uplo, n, d_A, n, d_W, d_work, lwork, devInfo);
    err = cudaDeviceSynchronize();
    //printf("stat: %d\n", cusolver_status);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == err);

    // copy eigenvalues
    cudaMemcpy(L1v, d_W, sizeof(double)*n, cudaMemcpyDeviceToHost);

    gsl_vector *res = gsl_vector_alloc(n);
    for(i = 0; i<n; i++)
    {
        gsl_vector_set(res, i, L1v[i]);
    }

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(devInfo);
    cudaFree(d_work);
    free(L1v);
    free(L1mat);
    cusolverDnDestroy(cusolverH);

    return res;

}

/* Computes the KL divergence between two density matrices.
 * Computes eigenvalues independently, sorts them, then 
 * computes divergence */
extern "C" double KL_divergence_cuda(gsl_matrix * L1, gsl_matrix * L2, double beta)
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

    /* compute eigenvalues */
    gsl_matrix *mats[2];
    size_t sizes[2];
    gsl_vector **ress;
    mats[0] = L1;
    mats[1] = L2;
    sizes[0] = n;
    sizes[1] = n;
    ress = cuda_batch_get_eigenvalues(mats, 2);
    /*  gsl_vector * L1v = cuda_get_eigenvalues(L1, n);
    gsl_vector * L2v = cuda_get_eigenvalues(L2, n);*/
    gsl_vector * L1v = ress[0];
    gsl_vector * L2v = ress[1];
    gsl_vector * rhov = gsl_vector_alloc(n);
    gsl_vector * sigmav = gsl_vector_alloc(n);

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
    gsl_vector_free(rhov);
    gsl_vector_free(sigmav);
    gsl_vector_free(L1v);
    gsl_vector_free(L2v);       
    return div;
}

/* Computes the KL divergence between two density matrices, 
 * that don't have to be same size
 * Computes eigenvalues independently, sorts them, then 
 * computes divergence */
extern "C" double KL_divergence_cuda_zeropad(gsl_matrix * L1, gsl_matrix * L2, double beta)
{
    double div = 0.0;
    double rval, sval;

    int i;
    size_t n1, n2, n_max;

    /* Check if they are square matrices and report size */
    if ((n1 = check_square_matrix(L1)) < 0) {
        printf("Rho matrix not Square!! \n");
        return 0;
    } else if ((n2 = check_square_matrix(L2)) < 0) {
        printf("Sigma matrix not Square \n");
        return 0;
    }
    n_max = (n1 > n2) ? n1 : n2;

    /* compute eigenvalues */
    gsl_matrix *mats[2];
    size_t sizes[2];
    gsl_vector **ress;

    mats[0] = L1;
    mats[1] = L2;
    sizes[0] = n1;
    sizes[1] = n2;
    ress = cuda_batch_get_eigenvalues(mats, 2);
    gsl_vector * L1v = ress[0];
    gsl_vector * L2v = ress[1];
    gsl_vector * rhov = gsl_vector_calloc(n_max);
    gsl_vector * sigmav = gsl_vector_calloc(n_max);

    /* Compute density eigenvalues */
    double r1, r2;
    double tr1=0, tr2=0;
    for (i = 0; i<n1; i++) {
        r1 = exp(beta*gsl_vector_get(L1v, i));
        gsl_vector_set(rhov, i, r1);
        tr1 += r1; 
    }

    for (i = 0; i<n2; i++) {
        r2 = exp(beta*gsl_vector_get(L2v, i));
        gsl_vector_set(sigmav, i, r2);
        tr2 += r2;
    }

    /* Sort eigenvalues */
    gsl_sort_vector(rhov);
    gsl_sort_vector(sigmav);

    /* Compute divergence */
    for (i = 0; i < n_max; i++) {
        rval = gsl_vector_get(rhov, i) / tr1;
        sval = gsl_vector_get(sigmav, i) / tr2;
        div += rval*(log(rval) - log(sval))/log(2.0);
    }
    /* Free Memory */
    gsl_vector_free(rhov);
    gsl_vector_free(sigmav);
    gsl_vector_free(L1v);
    gsl_vector_free(L2v);       
    free(ress);
    return div;
}
