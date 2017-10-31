/*
 * =====================================================================================
 *
 *       Filename:  hash_table.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/14/2017 12:30:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include "simplex.h"
#include "hash_table.h"

int ncollisions = 0;

struct simplex_hash_entry **get_empty_hash_table()
{
    /* Allocate space for hash table */
    struct simplex_hash_entry **table = malloc(NR_HASH *
            sizeof(struct simplex_hash_entry *));
    if (!table) {
        printf("Unable to allocate new hash table\n");
    }

    /* Allocate empty hash lists */
    int i;
    struct simplex_hash_entry *s_new;
    for (i = 0; i<NR_HASH; i++) {
        s_new = malloc(sizeof(struct simplex_hash_entry));
        s_new->s = NULL;
        s_new->next = NULL;
        table[i] = s_new;
    }
    
    return table;
}

struct simplex_hash_table * get_empty_hash_table_D()
{
    struct simplex_hash_table *out = calloc(1, sizeof(struct simplex_hash_table));

    if (!out) {
        printf("Unable to allocate table \n");
    }
    return out;
}

void free_hash_table(struct simplex_hash_entry **table) 
{
    int i;
    struct simplex_hash_entry *nx, *ny;

    for (i=0; i<NR_HASH; i++) {
        nx = table[i];
        while (nx != NULL) {
           ny = nx->next;
           free(nx);
           nx = ny; 
        }
    }
    free(table);
}

void free_hash_table_D(struct simplex_hash_table * table)
{
    free(table);
}

int check_hash(struct simplex_hash_entry **table, struct Simplex * sp)
{
    /* Returns 1 if simplex found in hash table, 0 otherwise */
    if (!sp) {
        return 1;
    }
    /* Compute simplex hash */
    unsigned int sp_hash = simplex_hash(sp);

    /* index into hash table */
    unsigned int index = sp_hash % NR_HASH;
    struct simplex_hash_entry *list = table[index];
    struct simplex_hash_entry *prev;

    /* check for presence of simplex */
    while (list != NULL) {
        if (list->s) {
            ncollisions++;
        }
        if (simplex_equals(sp, list->s)) {
            return 1;
        }
        prev = list;
        list = list->next;
    }
    /* not found, add to list */
    add_to_hash_list(prev, sp);
    return 0;
}

void add_to_hash_list(struct simplex_hash_entry *list, struct Simplex *sp)
{
    if (list->s == NULL) {
        /* Empty List */
        list->s = sp;
    }

    /* Run to end of list and insert */
    struct simplex_hash_entry *s_new = 
        malloc(sizeof(struct simplex_hash_entry));
    s_new->s = sp;
    s_new->next = NULL;
    while (list->next != NULL) {
        list = list->next;
    }
    list->next = s_new;
}

unsigned int simplex_hash(struct Simplex *s)
{
    /* Computes hash value for simplex s */
    int i;
    int dim = s->dim;
    unsigned int hc = dim + 1;
    for (i=0; i<=dim; i++) {
        hc = hc*314159 + s->vertices[i];
    }
    return hc;
}

int check_hash_D(struct simplex_hash_table *table, struct Simplex * sp)
{
    /* Linear probing.  Knuth V3 6.4 Alg. L */
    /* Returns 1 if simplex is in hash table */
    /* if not, it adds the simplex */
    if (!sp)
        return 1;

    /* Computes hash value for simplex s */
    int j;
    int dim = sp->dim;
    unsigned int hc = dim + 1;
    for (j=0; j<=dim; j++) {
        hc = hc*314159 + sp->vertices[j];
    }
    //unsigned int hash_p = simplex_hash(sp);
    //unsigned int index = hash_p % NR_HASH;
    unsigned int index = hc % NR_HASH;
    
    unsigned int i = index;
    while (i != index + 1) {
        if (!(table->table[i])) break;

        if (simplex_equals(table->table[i], sp)) return 1;

        if (i == 0) {
            i = NR_HASH - 1;
        } else {
            i -= 1;
        }
    }
    if (table->N == (NR_HASH-1)) {
        printf("Table Overflow!\n");
        return 0;
    } else {
        table->N++;
        table->table[i] = sp;
    }
    return 0;
}

