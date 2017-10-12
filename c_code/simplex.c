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
typedef struct simplex_list[MAXDIM] SCG;

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

Simplex *get_simplex_from_integer(unsigned int N)
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

int simplex_equals(Simplex * s1; Simplex * s2)
{
    /* Returns 1 if the two simplices are identical */
    
    /* easy case: dimensions not equal */
    if (s1->dim != s2->dim) {
        return 0;
    }

}

void scg_list_union(SCG * list1, SCG * list2, int dim)
{
    /*  form the union of scg lists */
    for ( dim = 0; dim < MAXDIM; dim++) {
        simplex_list * list1d = list1[dim];
        simplex_list * list2d = list2[dim];

        /*  Check and see if there is a null list */
        if ((list1d->s == NULL) && (list2d->s) == NULL) continue



    }
}

simplex_list * get_empty_simplex_list() 
{
    /*  returns an empty simplex list */
    simplex_list *out = malloc(sizeof(simplex_list));
    out->s = NULL;
    out->prev = NULL;
    out->next = NULL;
    return out;

}

void simplex_list_free(simplex_list *)
{
    /* TODO: Check for beginning of list */
    simplex_list * nx = sl->next;
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
        (*out)[dim] = get_empty_simplex_list();
    }
    return out;
}

void free_SCG(SCG * scg)
{
    for (int dim = 0; dim < MAXDIM; dim++) {
        simplex_list_free((*scg)[dim]);
    }
    free(scg);
}

void compute_chain_groups(Simplex * max_simps, int n_max_simps, SCG * scg_out)
{
    /* Compute the chain group generators for the complex
     * defined by the max_simps */

    /* Find the maximum dimension */
    int maxdim = 0;
    for (int i=0; i<n_max_simps; i++) {
        maxdim = max(max_simps[i]->dim, maxdim);
    }
    /* for each max simp, get the faces and add to the scg */
    for (int i=0; i<n_max_simps; i++) {
        SCG * faces = get_faces(max_simps[i]);
        for (int dim = 0; dim < MAXDIM; dim++) {
            /* take the union of face lists */
            scg_list_union(scg_out, faces, dim); 
        }
    }
    
}
