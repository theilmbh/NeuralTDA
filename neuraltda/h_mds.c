#include <math.h>
#include <complex.h>

double d_H(double complex x1, double complex x2)
{
	double a = cabs(x1 - x2)
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
	double tsq = (pow(v1, 2)+pow(v2, 2))/(pow(v3, 2) + pow(v4, 2)0;)

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
	double tsq = (pow(v1, 2)+pow(v2, 2))/(pow(v3, 2) + pow(v4, 2)0;)

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

double complex dE_dxa(double complex *X, double complex **data, double **w, int alpha, int q)
{

}

double complex HMDS_update(double complex *X, double complex **data, double **w, int alpha, int q)
{

}

double complex *fit_HMDS(double complex *X, double complex **data, double **w, double eta, double eps, int maxiter)
{

}