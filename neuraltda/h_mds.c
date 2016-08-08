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
	x = X[i];
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
	return dEdxa;
}

double complex HMDS_update(double complex *X, double complex **data, double **w, int alpha, int q)
{

	double *dE[2];
	dE[0] = -0.5*dE_dxa(X, data, w, alpha, n, 1);
	dE[1] = -0.5*dE_dxa(X, data, w, alpha, n, 2);

	double *alph_mat[2][2];
	

}

double complex *fit_HMDS(double complex *X, double complex **data, double **w, double eta, double eps, int maxiter)
{

}