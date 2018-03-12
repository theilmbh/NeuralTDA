/*
 * =====================================================================================
 *
 *       Filename:  binned_file.h
 *
 *    Description:  Definitions for the file header
 *
 *        Version:  1.0
 *        Created:  10/20/2017 09:41:22 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef BINNED_FILE_H
#define BINNED_FILE_H

#include <stdlib.h>
#include <stdtype.h>

#define BF_MAGIC 0xb4ad
#define NR_STIM_MAX 128;
#define STIM_NAME_MAX 128;

struct binned_file_header {
    uint16_t bf_magic;
    uint8_t n_stim;
    uint8_t n_cells;
    uint8_t n_win;
    uint8_t n_trial;
    char stim_names[NR_STIM_MAX][STIM_NAME_MAX];
};

#endif
