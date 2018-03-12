/*
 * =====================================================================================
 *
 *       Filename:  pyslsa_memcheck.c
 *
 *    Description:  Test for memory leaks
 *
 *        Version:  1.0
 *        Created:  10/30/2017 10:34:26 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include "simplex.h"
#include "slse.h"
#include "boundary_op.h"
#include <gsl/gsl_matrix.h>
#include <stdlib.h>

void run_computation()
{
    struct Simplex * s1 = create_empty_simplex();

    int i;
    int dim = 1;
    double beta = -0.15;
    for (i = 4; i < 6; i++) {
        add_vertex(s1, i);
    }


    SCG * scg1 = get_empty_SCG();

    struct Simplex * max_simps[2];
    max_simps[0] = s1;

    compute_chain_groups(max_simps, 1, scg1);
    print_SCG(scg1);
    
    free_simplex(s1);
    free_SCG(scg1);

}

int main(int argc, char **argv)
{
    run_computation();
    return 1;
}
