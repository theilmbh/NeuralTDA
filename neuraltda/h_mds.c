#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>


/* Function Declarations */
double d_H(double complex x1, double complex x2);
double ddH_dx1_1(double complex x1, double complex x2);
double ddH_dx1_2(double complex x1, double complex x2);
double complex mobius_xform(double complex z, double complex c, double theta);
double HMDS_E(double complex *X, double complex **data, double **w, int n);
double **get_distances(double complex *X, int n);
double complex dE_dxa(double complex *X, double complex **data, double **w, int alpha, int n, int q);
double complex HMDS_update(double complex *X, double complex **data, double **w, int n, int alpha, double eta, double lambda);
double complex *fit_HMDS(double complex *X, double complex **data, double **w, int n, double eta, double eps, int maxiter, int verbose);


double d_H(double complex x1, double complex x2)
{
	double a = cabs(x1 - x2);
	double b = cabs(1.0 - x1*conj(x2));
	return 2.0*catanh(a/b);
}

double ddH_dx1_1(double complex x1, double complex x2)
{
	
	double x11 = creal(x1);
	double x12 = cimag(x1);
	double x21 = creal(x2);
	double x22 = cimag(x2);

	double v1 = x11-x21;
	double v2 = x12-x22;

	double v3 = x11*x21 + x12*x22 - 1.0;
	double v4 = x11*x22 - x12*x21;
	double tsq = (pow(v1, 2)+pow(v2, 2))/(pow(v3, 2) + pow(v4, 2));

	double a = ((v1/(pow(v1, 2) + pow(v2,2))) - (x21*v3+x22*v4)/(pow(v3,2) + pow(v4,2)));
	double b = (2*sqrt(tsq))/(1-tsq);
	return b*a;

}

double ddH_dx1_2(double complex x1, double complex x2)
{
	double x11 = creal(x1);
	double x12 = cimag(x1);
	double x21 = creal(x2);
	double x22 = cimag(x2);

	double v1 = x11-x21;
	double v2 = x12-x22;

	double v3 = x11*x21 + x12*x22 - 1.0;
	double v4 = x11*x22 - x12*x21;
	double tsq = (pow(v1, 2)+pow(v2, 2))/(pow(v3, 2) + pow(v4, 2));

	double a = ((v2/(pow(v1, 2) + pow(v2,2))) - (x11*v4-x22*v3)/(pow(v3,2) + pow(v4,2)));
	double b = (2*sqrt(tsq))/(1-tsq);
	return b*a;
}

double complex mobius_xform(double complex z, double complex c, double theta)
{
	double complex a = theta*z +c;
	double complex b = conj(c)*theta*z +1;
	return a/b;
}

double HMDS_E(double complex *X, double complex **data, double **w, int n)
{
	double **distmat = get_distances(X, n);
	double E = 0.0;
	for (int i=0; i<n; i++)
	{
		for(int j=i+1; j<n; j++)
		{
			E += w[i][j]*pow(distmat[i][j]-data[i][j], 2);
		}
	}
	free(distmat);
	return E;
}

double **get_distances(double complex *X, int n)
{
	double **distmat = calloc(n*n, sizeof(double));
	double dist;
	for(int m=0; m<n; m++)
	{
		for(int k=m; k<n; k++)
		{
			dist = d_H(X[m], X[n]);
			distmat[m][n] = dist;
			distmat[n][m] = dist;
		}
	}
	return distmat;
}

double complex dE_dxa(double complex *X, double complex **data, double **w, int alpha, int n, int q)
{
	double **distmat = get_distances(X, n);
	double ddH;
	double complex x, y;

	double *dd_vec = calloc(n-alpha, sizeof(double));
	double dEdxa = 0.0;
	x = X[alpha];

	if(q==1)
	{
		for(int j=alpha+1; j<n; j++)
		{
			y = X[j];
			ddH = ddH_dx1_1(x, y);
			dEdxa += 2*w[alpha][j]*(distmat[alpha][j] - data[alpha][j])*ddH;
		}
	}
	else
	{
		for(int j=alpha+1; j<n; j++)
		{
			y = X[j];
			ddH = ddH_dx1_2(x, y);
			dEdxa += 2*w[alpha][j]*(distmat[alpha][j] - data[alpha][j])*ddH;
		}
	}
	free(dd_vec);
	free(distmat);
	return dEdxa;
}

double complex HMDS_update(double complex *X, double complex **data, double **w, int n, int alpha, double eta, double lambda)
{
	double lam;
	double dE[2];
	dE[0] = dE_dxa(X, data, w, alpha, n, 1);
	dE[1] = dE_dxa(X, data, w, alpha, n, 2);

	gsl_matrix *alpha_mat = gsl_matrix_calloc(2, 2);
	for(int a=0; a<2; a++)
	{
		for (int b=0; b<2; b++)
		{
			lam = 1;
			if(a==b)
			{
				lam = (1+lambda);
			}
			gsl_matrix_set(alpha_mat, a, b, dE[a]*dE[b]*lam);
		}
	}

	gsl_vector *beta = gsl_vector_calloc(2);
	for(int a=0; a<2; a++)
	{
		gsl_vector_set(beta, a, -0.5*dE[a]);
	}

	gsl_vector *delta_vec = gsl_vector_calloc(2);
	gsl_matrix *cov = gsl_matrix_calloc(2,2);
	double *chisq = malloc(sizeof(double));
	gsl_multifit_linear_workspace *worksp = gsl_multifit_linear_alloc(2, 2);
	gsl_multifit_linear(alpha_mat, beta, delta_vec, cov, chisq, worksp);


	double complex delta;
	delta = gsl_vector_get(delta_vec, 0) + I*gsl_vector_get(delta_vec, 1);
	double delmag = cabs(delta);

	if(eta > 1.0/delmag)
	{
		eta = 0.8*1.0/delmag;
	}

	double complex newpt = mobius_xform(X[alpha], eta*delta, 1);

	gsl_vector_free(delta_vec);
	gsl_vector_free(beta);
	gsl_matrix_free(alpha_mat);
	gsl_matrix_free(cov);
	gsl_multifit_linear_free(worksp);
	free(chisq);
	return newpt;
}

double complex *fit_HMDS(double complex *X, double complex **data, double **w, int n, double eta, double eps, int maxiter, int verbose)
{
	double diffp = 1;
	double lam, E, newE;
	double complex newpt, oldpt;
	double *lamm = malloc(n*sizeof(double));
	int iternum = 0;
	int a;
	srand(time(NULL));

	while(diffp > eps && iternum < maxiter)
	{
		a = rand() % n;
		lam = lamm[a];
		E = HMDS_E(X, data, w, n);
		newpt = HMDS_update(X, data, w, n, a, eta, lam);
		oldpt = X[a];
		X[a] = newpt;
		newE = HMDS_E(X, data, w, n);
		if((iternum % 50) == 0 && verbose)
		{
			printf("Iteration: %d;  E = %f", iternum, E);
		}
		if(newE>E)
		{
			lam = 10*lam;
		}
		else
		{
			lam = lam/10;
			X[a] = oldpt;
		}

		if(lam > 1e4)
		{
			lam = 1e4;
		}
		else if(lam < 1e-4)
		{
			lam = 1e-4;
		}
		diffp = fabs(newE - E);
		lamm[a] = lam;
		iternum += 1;
	}
	free(lamm);
	return X;
}

void read_data_distance_matrix(char *datafile, double **data_mat, int n)
{

}

void generate_initial_configuration(double complex *X, int n)
{
	double r, theta
	for(int i=0; i<n; i++)
	{
		r = (double)rand()/(double)RAND_MAX;
		theta=2*3.14159*(double)rand()/(double)RAND_MAX;
		X[i] = r*cexp(theta*I);
	}
}

void calculate_w(double **data, double **w int n)
{
	double Dsum = 0;

	for(int k=0; k<n; k++)
	{
		for(int l=k+1; l<n; l++)
		{
			Dsum += data[k][l];
		}
	}

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			if(i != j)
			{
				w[i][j] = 1.0/(Dsum*data[i][j]);
			}
			
		}
	}
}