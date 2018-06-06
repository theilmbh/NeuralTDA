/*
 * =====================================================================================
 *
 *       Filename:  slse.h
 *
 *    Description:  Simplicial laplacian spectral entropy
 *
 *        Version:  1.0
 *        Created:  10/18/2017 10:30:09 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef SLSE_H
#define SLSE_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

int check_square_matrix(gsl_matrix * a);
double KL_divergence(gsl_matrix * L1, gsl_matrix * L2, double beta);
double KL_divergence_cuda(gsl_matrix * L1, gsl_matrix * L2, double beta);
gsl_vector ** cuda_batch_get_eigenvalues(gsl_matrix ** mats, size_t n_scg);


#endif
