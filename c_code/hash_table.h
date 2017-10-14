/*
 * =====================================================================================
 *
 *       Filename:  hash_table.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/14/2017 12:26:31 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef HASTABLE_H
#define HASHTABLE_H

#include "simplex.h"

#define NR_HASH 2048
struct simplex_hash_entry {
    struct Simplex * s;
    struct simplex_hash_entry * next;
};

struct simplex_hash_entry **get_empty_hash_table();
void free_hash_table(struct simplex_hash_entry **table);
int check_hash(struct simplex_hash_entry **table, struct Simplex * sp);
void add_to_hash_list(struct simplex_hash_entry *list, struct Simplex *sp);
unsigned int simplex_hash(struct Simplex *s);

#endif
