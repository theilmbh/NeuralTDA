/*
 * =====================================================================================
 *
 *       Filename:  print.c
 *
 *    Description:  SLSA print routines
 *
 *        Version:  1.0
 *        Created:  06/08/2018 09:18:43 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>

#include <gsl/gsl_matrix.h>

#define TERM_WIDTH 12
#define TERM_HEIGHT 22

void print_matrix(gsl_matrix * mat) 
{
    int n1, n2;
    int nscreenx, nscreeny;
    int row_start, col_start;
    int row_end, col_end;
    int i, j;

    nscreenx = mat->size1 / TERM_WIDTH + 1;
    nscreeny = mat->size2 / TERM_HEIGHT + 1;

    for (n1 = 0; n1 < nscreenx; n1++) {
        for (n2 = 0; n2 < nscreeny; n2++) {
            row_start = n2*TERM_HEIGHT;
            row_end = (n2+1)*TERM_HEIGHT;
            col_start = n1*TERM_WIDTH;
            col_end = (n1+1)*TERM_WIDTH;
            printf("Matrix Elements (%d:%d, %d:%d)\n", 
                    row_start+1, row_end, col_start+1, col_end);
            for (i = row_start; i < row_end; i++) {
                for(j = col_start; j < col_end; j++) {
                    printf("%4.2f ", gsl_matrix_get(mat, i, j));
                }
                printf("\n");
            }
            printf("\n\n");
        }
    }
}
