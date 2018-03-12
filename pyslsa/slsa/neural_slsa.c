/*
 * =====================================================================================
 *
 *       Filename:  neural_slsa.c
 *
 *    Description:  Routines for processing neural data with SLSA
 *
 *        Version:  1.0
 *        Created:  10/20/2017 08:53:51 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "simplex.h"
#include "slse.h"
#include "boundary_op.h"

struct binned_file_header bfh;

SCG * binmat_to_scg(int binmat[][], int ncells, int nwin)
{
    /* Computes the SCG assocated to a binary matrix */
    int i, j;
    struct Simplex * new_max_simplex;

    /* Allocate max simplex list */
    struct Simplex * max_simps[] = malloc(nwin*sizeof(struct Simplex *));
    
    /* For each window we compute the cell group */
    for (i = 0; i < nwin; i++) {
        new_max_simplex = get_empty_simplex();
        for (j = 0; j < ncells; j++) {
            if (binmat[j][i]) {
                add_vertex(new_max_simplex, j);
            }
        }
        max_simps[i] = new_max_simplex;
    }

    /* Compute SCG */
    SCG * out = get_empty_SCG();
    compute_chain_groups(max_simps, nwin, out);
    return out;
}

FILE * open_binned_file(char *filename)
{
    /* try to open the file */
    FILE * fd;
    fd = fopen(filename, "rb");
    if (!fd) {
        printf("unable to open binned file: %s\n", filename);
    }

    /* Read header */
    fread(&bfh, 1, sizeof(struct binned_file_header), fd);
    if (bfh.bf_magic != BF_MAGIC) {
        printf("Not a valid binned data file\n");
    }
}
char **get_stimuli_names(FILE * bf, int * nstim)
{
    hid_t file;
    hid_t grp;
    herr_t status;
    hsize_t n_stim;

    /* open file */
    file = H5Fopen(bf_name, H5F_ACC_RD, H5P_DEFAULT);
    grp = H5Gopen(file, "/", H5P_DEFAULT);

    err = H5Gget_num_objs(grp, &n_stim);
    out = malloc(n_stim*sizeof(char *));
    for (int i = 0; i < n_stim; i++) {
        out[i] = malloc(MAX_NAME*sizeof(char));
        H5Gget_objname_by_idx(gid, (hsize_t)i, out[i], (size_t)MAX_NAME);
    }

    *nstim = (int)n_stim;
    return out; 
}

void get_population_tensor_shape(char * bf_name, int * n_cells,
                                 int * n_win, int * n_trial)
{
    int num_attrs;
    hid_t aid;
    int i;
    na = 
}
