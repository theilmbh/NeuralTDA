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


#define MAXNAME 128
#define MAXMS 12
#define MAXDIM 10

struct Simplex {
    unsigned int vertices[MAXDIM];
    int dim;
};

struct simplex_list {
    struct Simplex * s;
    struct simplex_list * next;
    struct simplex_list * prev;
};

/* Array of simplex lists forms simplicial complex generators */
typedef struct SCG {
    struct simplex_list *x[MAXDIM];
} SCG;

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

void get_faces_common(simplex_list_t face_list, unsigned int N, int n_verts)
{
    unsigned int k, dim;

    for (k = 0; k < (1 << n_verts); k++) {
        if ((k & N) == k) {
            dim = num_ones(k) - 1;
            for(int i = 0; i < n_verts; i++) {
                if (check_bit(k, i)) {
                    
                }
            } 
        }
    }
}

int int_cmp(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

int simplex_equals(struct Simplex * s1, struct Simplex * s2)
{
    /* Returns 1 if the two simplices are identical */
    
    /* easy case: dimensions not equal */
    if (s1->dim != s2->dim) {
        return 0;
    }
    
    /* Sort the vertices */
    qsort(s1->vertices, s1->dim+1, sizeof(int), int_cmp);
    qsort(s2->vertices, s2->dim+1, sizeof(int), int_cmp);

    /* Check if arrays are equal */
    for (int i = 0; i <= s1->dim; i++) {
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
    /*  This is ugly! */
    for ( dim = 0; dim < MAXDIM; dim++) {
        struct simplex_list * list1d = scg1->x[dim];
        struct simplex_list * list2d = scg2->x[dim];

        /*  Check and see if there is a null list */
        if ((list1d->s == NULL) && (list2d->s) == NULL) continue;

        struct simplex_list * l1 = list1d;
        struct simplex_list * l2 = list2d;
        struct simplex_list * l2start = list2d; /* Original start of list2 */
        struct simplex_list * c = NULL;        /* Placeholder */
        int do_add; /* Flag to decide whether to add or not */
        while (l1 != NULL) {
            do_add = 1;
            while (l2 != NULL) {
                if (simplex_equals(l1->s, l2->s)) {
                    do_add = 0;
                    break;
                }
                l2 = l2->next;
            }
            if (do_add) {
                /* simplex not found, add to beginning of list2 */
                list2d->prev = l1;
                c = l1;
                l1 = l1->next; /* Advance l1 */
                c->next = list2d; 
                c->prev = NULL; /* Make it the head of the list */
                list2d = c;  /* Advance head of list2d */
            }
            l2 = l2start; /* reset */
        }
        scg2->x[dim] = list2d; /* update SCG */
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

SCG * get_empty_SCG() 
{
    SCG * out = malloc(sizeof(SCG));
    for (int dim = 0; dim < MAXDIM; dim++)
    {
        out->x[dim] = get_empty_simplex_list();
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

void compute_chain_groups(struct Simplex * max_simps,
                          int n_max_simps, SCG * scg_out)
{
    /* Compute the chain group generators for the complex
     * defined by the max_simps */

    /* Find the maximum dimension */
    int maxdim = 0;
    for (int i=0; i<n_max_simps; i++) {
        maxdim = max_simps[i].dim < maxdim ? maxdim : max_simps[i].dim;
    }
    /* for each max simp, get the faces and add to the scg */
    for (int i=0; i<n_max_simps; i++) {
        SCG * faces = get_faces(max_simps[i]);
        for (int dim = 0; dim < MAXDIM; dim++) {
            /* take the union of face lists */
            scg_list_union(faces, scg_out, dim); 
        }
    }
    
}
