#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>


/* Function Declarations */
double d_H(double complex x1, double complex x2);
double ddH_dx1_1(double complex x1, double complex x2);
double ddH_dx1_2(double complex x1, double complex x2);
double complex mobius_xform(double complex z, double complex c, double theta);
double HMDS_E(double complex *X, double *data, double *w, int n);
double *get_distances(double complex *X, int n);
double complex dE_dxa(double complex *X, double *data, double *w, int alpha, int n, int q);
double complex HMDS_update(double complex *X, double *data, double *w, int n, int alpha, double eta, double lambda);
double complex *fit_HMDS(double complex *X, double *data, double *w, int n, double eta, double eps, int maxiter, int verbose);


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

double HMDS_E(double complex *X, double *data, double *w, int n)
{
    double *distmat = get_distances(X, n);
    double E = 0.0;
    for (int i=0; i<n; i++)
    {
        for(int j=i+1; j<n; j++)
        {
            E += w[i*n+j]*pow(distmat[i*n+j]-data[i*n+j], 2);
        }
    }
    free(distmat);
    return E;
}

double *get_distances(double complex *X, int n)
{
    double *distmat = calloc(n*n, sizeof(double));
    double dist;
    for(int m=0; m<n; m++)
    {
        for(int k=m; k<n; k++)
        {
            dist = d_H(X[m], X[k]);
            distmat[m*n+k] = dist;
            distmat[k*n+m] = dist;
        }
    }
    return distmat;
}

double complex dE_dxa(double complex *X, double *data, double *w, int alpha, int n, int q)
{
    double *distmat = get_distances(X, n);
    double ddH;
    double complex x, y;

    double *dd_vec = calloc(n, sizeof(double));
    double dEdxa = 0.0;
    x = X[alpha];

    if(q==1)
    {
        for(int j=0; j<n; j++)
        {
            if(j!=alpha)
            {
                y = X[j];
                ddH = ddH_dx1_1(x, y);
                dEdxa += w[alpha*n+j]*(distmat[alpha*n+j] - data[alpha*n+j])*ddH;
            }
        }
    }
    else
    {
        for(int j=0; j<n; j++)
        {
            if(j!=alpha)
            {
                y = X[j];
                ddH = ddH_dx1_2(x, y);
                dEdxa += w[alpha*n+j]*(distmat[alpha*n+j] - data[alpha*n+j])*ddH;
            }

        }
    }
    free(dd_vec);
    free(distmat);
    return dEdxa;
}

double complex HMDS_update(double complex *X, double *data, double *w, int n, int alpha, double eta, double lambda)
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

double complex *fit_HMDS(double complex *X, double *data, double *w, int n, double eta, double eps, int maxiter, int verbose)
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
        diffp = fabs(newE - E);

        /*if(a==n-1)
        {
            printf("Old n-1: %f, %f\n", creal(oldpt), cimag(oldpt));
            printf("new n-1: %f, %f\n", creal(newpt), cimag(newpt));
        }*/

        if((iternum % 50) == 0 && verbose)
        {
            printf("Iteration: %d;  E = %f\n", iternum, E);
        }
        if(newE>=E)
        {
            lam = 10*lam;
            X[a] = oldpt;
            diffp=1;
        }
        else
        {
            lam = lam/10;

        }

        if(lam > 1e4)
        {
            lam = 1e4;
        }
        else if(lam < 1e-4)
        {
            lam = 1e-4;
        }

        
        lamm[a] = lam;
        iternum += 1;
    }
    printf("Finished with %d iterations, E= %f\n", iternum, E);
    free(lamm);
    return X;
}

void read_data_distance_matrix(char *data_filename, double *data_mat, int n)
{
    FILE *data_file;
    data_file = fopen(data_filename, "r");
    if(!data_file)
    {
        printf("Error Opening Datafile!\n");
        exit(-1);
    }

    size_t nread = fread(data_mat, sizeof(double), n*n, data_file);
    if(!nread)
    {
        printf("Error Reading From File!\n");
    }
    fclose(data_file);
}

void save_embedding(double complex *X, int n, char *embed_filename)
{
    FILE *embed_file;
    embed_file = fopen(embed_filename, "w");
    size_t nwrite = fwrite(X, sizeof(double complex), n, embed_file);
    if(!nwrite)
    {
        printf("Error Writing to File\n");
    }
}

void generate_initial_configuration(double complex *X, int n)
{
    double r, theta;
    for(int i=0; i<n; i++)
    {
        r = (double)rand()/(double)RAND_MAX;
        theta=2*3.14159*(double)rand()/(double)RAND_MAX;
        X[i] = r*cexp(theta*I);
    }
}

void calculate_w(double *data, double *w, int n)
{
    double Dsum = 0;

    for(int k=0; k<n; k++)
    {
        for(int l=k+1; l<n; l++)
        {
            Dsum += data[k*n+l];
        }
    }

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(i != j)
            {
                w[n*i+j] = 1.0/(Dsum*data[n*i+j]);
            } 
        }
    }
}

void print_embedding(double complex *X, int n)
{
    for(int i=0; i<n; i++)
    {
        printf("Point %d:   %f, %f \n", i, creal(X[i]), cimag(X[i]));
    }
}

void print_distance_matrix(double *dist_mat, int n)
{
    printf("\nDistance Matrix\n");
    printf("===============\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            printf("%+.3f  ", dist_mat[i*n+j]);
        }
        printf("\n");
    }
    printf("===============\n\n");
}

void run_HMDS(char *data_filename, char *embed_filename, int n, double eta, double eps, int maxiter, int verbose)
{

    /* Allocate Arrays */
    double complex *X = malloc(n*sizeof(double complex));
    double *w = malloc(n*n*sizeof(double));
    double *data = malloc(n*n*sizeof(double));

    read_data_distance_matrix(data_filename, data, n);
    generate_initial_configuration(X, n);
    calculate_w(data, w, n);
    fit_HMDS(X, data, w, n, eta, eps, maxiter, verbose);
    save_embedding(X, n, embed_filename);

    free(X);
    free(w);
    free(data);

}

void test_HMDS(int n)
{

    printf("Testing HMDS\n");

    char *embed_filename = "/Users/brad/test_hmds.dat";
    double *test_dist_mat = calloc(n*n, sizeof(double));
    double *final_dist_mat;
    double rand_dist;

    printf("Creating Distance Matrix\n");
    for(int i=0; i<n; i++)
    {
        for(int j=i+1; j<n; j++)
        {
            rand_dist = 5*((double)rand()/(double)RAND_MAX);
            test_dist_mat[n*i+j] = rand_dist;
            test_dist_mat[n*j+i] = rand_dist;
        }
    }
    printf("Allocating arrays\n");
    double complex *X = malloc(n*sizeof(double complex));
    double *w = malloc(n*n*sizeof(double));
    
    printf("Generating Initial Configuration\n");
    generate_initial_configuration(X, n);
    printf("Calculating w\n");
    calculate_w(test_dist_mat, w, n);

    int verbose=1;
    double eps = 1e-6;
    double eta=0.3;
    int maxiter=2000;

    print_embedding(X, n);
    fit_HMDS(X, test_dist_mat, w, n, eta, eps, maxiter, verbose);
    final_dist_mat = get_distances(X, n);
    save_embedding(X, n, embed_filename);
    print_embedding(X, n);
    print_distance_matrix(test_dist_mat, n);
    print_distance_matrix(final_dist_mat, n);

    free(X);
    free(w);
    free(test_dist_mat);

}

int main(int argc, char **argv)
{
    srand(time(NULL));

    char *input_file = NULL;
    char *output_file = NULL;
    double eps, eta;
    int verbose, maxiter, n;
    int c;

    while((c = getopt(argc, argv, "i:o:e:n:m:h:v")) != -1)
        switch(c)
        {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'e':
                eps = atof(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'm':
                maxiter = atoi(optarg);
                break;
            case 'h':
                eta = atof(optarg);
                break;
            case 'v':
                verbose = 1;
                break;
            default:
                printf("Invalid arguments\n");
                abort();

        }

    run_HMDS(input_file, output_file, n, eta, eps, maxiter, verbose);

    return 0;

}