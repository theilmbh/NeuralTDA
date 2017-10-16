/*
 * ============================================================================
 *
 *       Filename:  simplex.c
 *
 *    Description:  Routines for manipulating simplicial complexes
 *
 *        Version:  1.0
 *        Created:  09/20/2017 06:38:26 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "simplex.h"
#include "hash_table.h" 

unsigned int num_ones(unsigned int N)
{
    unsigned int count = 0;

    while ( N > 0 ) {
        N = N & (N-1);
        count++;
    }

    return count;
}

int check_bit(unsigned int N, unsigned int i)
{
    if ( N & (1 << i) ) {
        return 1;
    } else {
        return 0;
    }
}

struct Simplex *get_simplex_from_integer(unsigned int N)
{
    /* Returns a pointer to a simplex struct
     * built from the integer N */
    int dim = num_ones(N) - 1;
    struct Simplex *out = malloc(sizeof(struct Simplex));
    out->dim = dim;
    return out;
}

void get_faces_common(unsigned int N, int *verts, int dim, SCG * scg_temp)
{
    struct Simplex * s_new;
   for (int k = 0; k <= N; k++) {
       if ( (k & N) == k ) {
            s_new = create_empty_simplex();
            for ( int j = 0; j < dim+1; j++) {
                if (check_bit(k, j)) {
                    add_vertex(s_new, verts[j]);
                }
            }
            scg_add_simplex_nocheck(scg_temp, s_new);
       }
   } 
}

unsigned int integer_from_simplex(struct Simplex * simp)
{
    unsigned int N;
    N = (1 << (simp->dim + 1)) - 1;
    return N;
}

SCG * get_faces(struct Simplex * simp)
{
    SCG * out = get_empty_SCG();
    int N;

    N = integer_from_simplex(simp);
    get_faces_common(N, simp->vertices, simp->dim, out);
    return out;
}

int int_cmp(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

void free_simplex(struct Simplex *s)
{
    free(s);
}

struct Simplex * create_empty_simplex()
{
    struct Simplex * s_out = malloc(sizeof(struct Simplex));
    for (int i=0; i < MAXDIM; i++) {
        s_out->vertices[i] = -1;
    }
    s_out->dim = -1;
    return s_out;
}

void add_vertex(struct Simplex * s, int v)
{
    if (s->dim == MAXDIM) return;
    s->dim++;
    s->vertices[s->dim] = v;
    qsort(s->vertices, s->dim+1, sizeof(int), int_cmp);
}

struct Simplex * create_simplex(unsigned int *vertices, int dim)
{
    struct Simplex * s_out = malloc(sizeof(struct Simplex));
    memcpy(s_out->vertices, vertices, MAXDIM*sizeof(unsigned int));
    s_out->dim = dim;
    return s_out;
}

int simplex_equals(struct Simplex * s1, struct Simplex * s2)
{
    /* Returns 1 if the two simplices are identical */
    if ((!s1) || (!s2)) {
        return 0;
    } 

    /* easy case: dimensions not equal */
    if (s1->dim != s2->dim) {
        return 0;
    }
    
    /* Sort the vertices */
    qsort(s1->vertices, s1->dim+1, sizeof(int), int_cmp);
    qsort(s2->vertices, s2->dim+1, sizeof(int), int_cmp);

    /* Check if arrays are equal */
    for (int i = 0; i <= MAXDIM; i++) {
        if (s1->vertices[i] != s2->vertices[i]) {
            return 0;
        }
    }
    /* if we make it here, they are the same */
    return 1;
}

void scg_list_union(SCG * scg1, SCG * scg2)
{
    /*  form the union of scg lists */
    /*  The output is in list2 */
    int dim;
    for ( dim = 0; dim < MAXDIM; dim++) {
        struct simplex_list * list1d = scg1->x[dim];
        struct simplex_list * list2d = scg2->x[dim];

        /*  Check and see if there is a null list */
        if ((list1d->s == NULL) && (list2d->s) == NULL) continue;

        struct simplex_list * l1 = list1d;
        struct simplex_list * l2 = list2d;

        /* Add each simplex from l1 to l2 */
        while (l1 != NULL) {
            add_simplex(l2, l1->s);
            l1 = l1->next;
        }
    }
}

void scg_list_union_hash(SCG * scg1, SCG * scg2,
                         struct simplex_hash_table *table2)
{
    int dim;
    struct simplex_hash_table * table;
    for (dim = 0; dim < MAXDIM; dim++) {
        struct simplex_list * list1d = scg1->x[dim];
        struct simplex_list * list2d = scg2->x[dim];

        /*  Check and see if there is a null list */
        if ((list1d->s == NULL) && (list2d->s) == NULL) continue;

        struct simplex_list * l1 = list1d;
        struct simplex_list * l2 = list2d;
        
        table = get_empty_hash_table_D();
        
        /* add list1 to temp list */
        while (l2 != NULL) {
            if (l2->s) {
                check_hash_D(table, l2->s);
            }
            l2 = l2->next;
        }
       
        /* add list2 to temp list, checking hash table */ 
        while (l1 != NULL) {
            if (l1->s && !check_hash_D(table, l1->s)) {
                add_simplex_nocheck(list2d, l1->s);
                scg2->cg_dim[dim]++; /* Increment Chain Group Dimension */
            }
            l1 = l1->next;
        }
        free_hash_table_D(table);
    }
}

struct simplex_list * get_empty_simplex_list() 
{
    /*  returns an empty simplex list */
    struct simplex_list *out = malloc(sizeof(struct simplex_list));
    out->s = NULL;
    out->prev = NULL;
    out->next = NULL;
    return out;

}

void simplex_list_free(struct simplex_list * sl)
{
    /* TODO: Check for beginning of list */
    struct simplex_list * nx = sl->next;
    while(nx) {
        free(sl);
        sl = nx;
        nx = sl->next;
    }
}

/* Checks if a simplex is already in a simplex list */
/* If so, returns a pointer to the list entry */
struct simplex_list * simplex_list_isin(struct simplex_list *slist,
                                        struct Simplex *s)
{
    struct simplex_list * retcode = NULL;
    while ( slist != NULL ) {
        if (simplex_equals(slist->s, s)) {
            retcode = slist;
            return retcode;
        }
        slist = slist->next;
    }
    return retcode;
}

struct simplex_list * remove_simplex(struct simplex_list *slist,
                                     struct Simplex *s)
{
    struct simplex_list *entry = simplex_list_isin(slist, s);
    struct simplex_list *ret = slist;
    if (entry != NULL) {
        if (entry->prev != NULL) {
            /* Entry is not first in list */
            entry->prev->next = entry->next;
        }
        if (entry->next != NULL) {
            /* Entry is not last in list */
            entry->next->prev = entry->prev;
        }
        if (entry->prev == NULL) {
            /* Entry is first in list */
            ret = entry->next;
        }
    }
    return ret;
}

void add_simplex_nocheck(struct simplex_list *slist, struct Simplex *s)
{
    if (!slist->s) {
            slist->s = s;
            return;
    }

    /* Add to second spot in list */
    if (slist != NULL) {
        struct simplex_list *s_new = get_empty_simplex_list();
        struct simplex_list *eltwo = slist->next;

        s_new->s = s;
        slist->next = s_new;
        s_new->prev = slist;
        s_new->next = eltwo;
        if (eltwo) {
            eltwo->prev = s_new;
        }
    }
    return;
}

/* Adds a simplex to a simplex list if not already present */
void add_simplex(struct simplex_list *slist, struct Simplex *s)
{
    if (slist->s == NULL) {
        /* Empty list */
        slist->s = s;
        return;
    }

    if (!simplex_list_isin(slist, s)) {
        struct simplex_list *new = get_empty_simplex_list();
        new->s = s;
        /* Advance to end of list */
        while (slist->next != NULL) {
            slist = slist->next;
        }
        /* Add new simplex */
        slist->next = new;
        new->prev = slist;
    }
}

SCG * get_empty_SCG() 
{
    SCG * out = malloc(sizeof(SCG));
    for (int dim = 0; dim < MAXDIM; dim++)
    {
        out->x[dim] = get_empty_simplex_list();
        out->cg_dim[dim] = 0;
    }
    return out;
}

void free_SCG(SCG * scg)
{
    for (int dim = 0; dim < MAXDIM; dim++) {
        simplex_list_free(scg->x[dim]);
    }
    free(scg);
}

void scg_add_simplex(SCG * scg, struct Simplex * s)
{
    int d = s->dim;
    if (d >= 0) {
        add_simplex(scg->x[d], s);
    }
}

void scg_add_simplex_nocheck(SCG * scg, struct Simplex * s)
{
    int d = s->dim;
    if (d >= 0) {
        add_simplex_nocheck(scg->x[d], s);
        scg->cg_dim[d]++;
    }
}

/* print functions */
void print_simplex(struct Simplex * s)
{
    if (s == NULL) {
        return;
    }

    printf("SIMPLEX | D = %d | Vertices: ", s->dim);
    
    for (int i = 0; i < MAXDIM; i++)
    {
        if (s->vertices[i] >= 0) {
            printf("%d, ", s->vertices[i]);
        }
    }
    printf("\n");
}

void print_simplex_list(struct simplex_list *sl)
{
    if (sl->s == NULL) return;

    printf("SIMPLEX LIST\n");
    printf("------------\n");

    while (sl != NULL)
    {
        print_simplex(sl->s);
        sl = sl->next;
    }
    printf("\n");
}

void print_SCG(SCG * scg)
{
    printf("SCG\n");
    printf("---\n");
    for (int i = 0; i < MAXDIM; i++) {
        if (scg->x[i]->s != NULL) {
            printf("Dimension %d\n", i);
            print_simplex_list(scg->x[i]);
        }
    }
    printf("\n");
}

void compute_chain_groups(struct Simplex ** max_simps,
                          int n_max_simps, SCG * scg_out)
{
    /* Compute the chain group generators for the complex
     * defined by the max_simps */

    /* Find the maximum dimension */
    int maxdim = 0;
    for (int i=0; i<n_max_simps; i++) {
        maxdim = max_simps[i]->dim < maxdim ? maxdim : max_simps[i]->dim;
    }

    /* for each max simp, get the faces and add to the scg */
    struct simplex_hash_table *table = get_empty_hash_table_D();
    for (int i=0; i<n_max_simps; i++) {
        SCG * faces = get_faces(max_simps[i]);
        /* take the union of face lists */
        scg_list_union_hash(faces, scg_out, table); 
    }
    
}
