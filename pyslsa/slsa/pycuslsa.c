/*
 * =============================================================================
 *
 *       Filename:  pyslsa.c
 *
 *    Description:  PyCuSLSA (Py salsa)  CUDA-Accelerated 
 *                  Python simplicial laplacian spectral 
 *                  analysis.
 *
 *        Version:  1.0
 *        Created:  10/20/2017 01:28:22 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  Gentner Lab
 *
 * =============================================================================
 */

#include <Python.h>
#include <string.h>

#include "simplex.h"
#include "boundary_op.h"
#include "slse.h"

/* 
 *  Python Simplex object definition
 *  A python simplex consists of simply a simplex as defined in simplex.c
 */
typedef struct {
	PyObject_HEAD struct Simplex *s;

} pyslsa_SimplexObject;

/*  
 *  Create an empty simplex and return it wrapped in a python Simplex object
 */
static PyObject *Simplex_new(PyTypeObject * type,
			     PyObject * args, PyObject * kwds)
{
	pyslsa_SimplexObject *self;
	self = (pyslsa_SimplexObject *) type->tp_alloc(type, 0);
	if (self != NULL) {
		self->s = create_empty_simplex();

	}
	return (PyObject *) self;
}

/*
 *  Destroy a python simplex by calling the appropriate "free" method
 *  as defined in the pyslsa_SimplexType structure
 */
static void Simplex_dealloc(pyslsa_SimplexObject * self)
{
	Py_TYPE(self)->tp_free((PyObject *) self);
}

/*
 *  Free the C simplex stored in the python simplex object
 */
static void Simplex_free(pyslsa_SimplexObject * self)
{
	free_simplex(self->s);
}

/*
 *  Add a vertex to a simplex, increasing its dimension.  
 *  This function implements the add_vertex method called from Python
 *  It expects an integer label for the vertex
 */
static PyObject *Simplex_add_vertex(pyslsa_SimplexObject * self,
				    PyObject * args)
{
	int i;
	if (!PyArg_ParseTuple(args, "i", &i)) {
		return NULL;
	}
	add_vertex(self->s, i);
	Py_RETURN_NONE;
}

/*
 *  Return the dimension of the simplex as an integer, 
 *  namely, the number of vertices - 1
 */
static PyObject *Simplex_dimension(pyslsa_SimplexObject * self)
{
	return Py_BuildValue("i", self->s->dim);
}

/*
 *  Define the methods available from Python to manipulate simplex objects
 */
static PyMethodDef Simplex_methods[] = {
	{"add_vertex", (PyCFunction) Simplex_add_vertex, METH_VARARGS,
	 "Add a vertex to the simplex"},
	{"get_dim", (PyCFunction) Simplex_dimension,
	 METH_NOARGS, "Get Simplex Dimension"},
	{NULL}
};

/*
 *  Define the Python Simplex type
 */
static PyTypeObject pyslsa_SimplexType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	    "pyslsa.Simplex",
	sizeof(pyslsa_SimplexObject),
	0,
	(destructor) Simplex_dealloc,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	"Simplex objects",
	0,
	0,
	0,
	0,
	0,
	0,
	Simplex_methods,	/* tp_methods */
	0,			/* tp_members */
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	Simplex_new,
	(freefunc) Simplex_free,
};

/* ************************************************************************* */
/* Simplicial Complex Object Definitions                                     */
/* ************************************************************************* */

/*
 *  Define the Simplicial Complex (SCG - "Simplicial Chain Group") python object
 *  A Python SCG consists of a C SCG wrapped in python stuff
 */
typedef struct {
	PyObject_HEAD SCG * scg;
	gsl_matrix *L1;
} pyslsa_SCGObject;

/*
 *  Destroy a Python SCG object
 */
static void SCG_dealloc(pyslsa_SCGObject * self)
{
	Py_TYPE(self)->tp_free((PyObject *) self);
}

/*
 *  Free the SCG stored in a Python SCG object
 */
static void SCG_free(pyslsa_SCGObject * self)
{
	free_SCG(self->scg);
}

/*
 *  Create a new Python SCG object by creating an empty C SCG
 */
static PyObject *SCG_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
	pyslsa_SCGObject *self;
	self = (pyslsa_SCGObject *) type->tp_alloc(type, 0);
	if (self != NULL) {
		self->scg = get_empty_SCG();
		self->L1 = NULL;

	}
	return (PyObject *) self;
}

/*
 * Adds a top-level simplex to the SCG, recomputing the chain groups of each
 * dimension
 */
static PyObject *PySCG_add_max_simplex(pyslsa_SCGObject * self,
				       PyObject * args, PyObject * kwds)
{
	pyslsa_SimplexObject *maxsimp;
	if (!PyArg_ParseTuple(args, "O", &maxsimp))
		return NULL;

	scg_add_max_simplex(self->scg, maxsimp->s);
	Py_RETURN_NONE;
}

/*
 *  Disply the internal structure of the SCG by 
 *  printing the generators in each dimension
 */
static PyObject *PySCG_print(pyslsa_SCGObject * self)
{
	print_SCG(self->scg);
	Py_RETURN_NONE;
}

/*
 *  Print the matrix of the simplicial laplacian in dimension d
 *  *** Need to check for bounds on dimension! ***
 */
static PyObject *PySCG_print_laplacian(pyslsa_SCGObject * self, PyObject * args)
{
	int d;
	if (!PyArg_ParseTuple(args, "i", &d))
		return NULL;

	int Ldim = self->scg->cg_dim[d];
	gsl_matrix *Lap = compute_simplicial_laplacian(self->scg, d);
	for (int i = 0; i < Ldim; i++) {
		for (int j = 0; j < Ldim; j++) {
			printf("%f ", gsl_matrix_get(Lap, i, j));
		}
		printf("\n");
	}
	Py_RETURN_NONE;
}

/*
 *  Return the size of the laplacian matrix (dim x dim) in SCG dimension d
 *  *** Check Dimension Bounds ***
 */
static PyObject *PySCG_get_laplacian_dim(pyslsa_SCGObject * self,
					 PyObject * args)
{
	int d;
	if (!PyArg_ParseTuple(args, "i", &d))
		return NULL;

	int dim = self->scg->cg_dim[d];
	return Py_BuildValue("i", dim);
}

/*
 *  Print the matrix associated with the SCG boundary operator in dimension d
 */
static PyObject *PySCG_print_boundary_op(pyslsa_SCGObject * self,
					 PyObject * args)
{
	int d;
	if (!PyArg_ParseTuple(args, "i", &d))
		return NULL;

	int Ddimr = self->scg->cg_dim[d - 1];
	int Ddimc = self->scg->cg_dim[d];
	gsl_matrix *D = compute_boundary_operator_matrix(self->scg, d);
	for (int i = 0; i < Ddimr; i++) {
		for (int j = 0; j < Ddimc; j++) {
			printf("%f ", gsl_matrix_get(D, i, j));
		}
		printf("\n");
	}
	Py_RETURN_NONE;
}

/*
 *  Python methods available for manipulating SCG objects
 */
static PyMethodDef SCG_methods[] = {
	{"add_max_simplex", (PyCFunction) PySCG_add_max_simplex, METH_VARARGS,
	 "Adds a top-level simplex to the simplicial complex"},
	{"print", (PyCFunction) PySCG_print, METH_NOARGS,
	 "Print the simplicial complex"},
	{"print_L", (PyCFunction) PySCG_print_laplacian, METH_VARARGS,
	 "Print the simplicial laplacian of dimension d"},
	{"print_D", (PyCFunction) PySCG_print_boundary_op, METH_VARARGS,
	 "Print the boundary operator of dimension d"},
	{"L_dim", (PyCFunction) PySCG_get_laplacian_dim, METH_VARARGS,
	 "Print the dimension of the d-Laplacian matrix"},
	{NULL}
};

/*
 *  Definition of the Simplicial Complex (SCG) Python type
 */
static PyTypeObject pyslsa_SCGType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	    "pyslsa.SCG",
	sizeof(pyslsa_SCGType),
	0,
	(destructor) SCG_dealloc,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	"Simplicial complex objects",
	0,
	0,
	0,
	0,
	0,
	0,
	SCG_methods,		/* tp_methods */
	0,			/* tp_members */
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	SCG_new,
	(freefunc) SCG_free,
};

/* ************************************************************************* */
/* PySLSA Module Definitions                                                 */
/* ************************************************************************* */

static char pyslsa_docs[] =
    "PyCuSLSA: CUDA-Accelerated Simplicial Laplacian Spectral Analyzer";

/*
 *  Python method to build a simplicial complex from a list of top-level (max)
 *  simplices
 */
static PyObject *build_SCG(PyObject * self, PyObject * args)
{
	Py_ssize_t ind, vert_ind;
	PyObject *max_simps;
	PyObject *simp_verts;
	struct Simplex *new_sp;
	pyslsa_SCGObject *out;

	if (!PyArg_ParseTuple(args, "O", &max_simps))
		return NULL;

	int n_max_simp = PyList_Size(max_simps);

	/* Get a new SCG and allocate max simp list */
	out = (pyslsa_SCGObject *) SCG_new(&pyslsa_SCGType, NULL, NULL);
	struct Simplex **max_simp_list = malloc(n_max_simp *
						sizeof(struct Simplex *));

	for (ind = 0; ind < n_max_simp; ind++) {
		new_sp = create_empty_simplex();
		simp_verts = PyList_GetItem(max_simps, ind);

		for (vert_ind = 0; vert_ind < PyTuple_Size(simp_verts);
		     vert_ind++) {
			add_vertex(new_sp,
				   (int)
				   PyLong_AsLong(PyTuple_GetItem
						 (simp_verts, vert_ind)));
		}
		max_simp_list[ind] = new_sp;
	}

	compute_chain_groups(max_simp_list, n_max_simp, out->scg);

	free(max_simp_list);
	out->L1 = compute_simplicial_laplacian(out->scg, 1);
	return (PyObject *) out;
}

/*
 *  Compute the union of two python simplicial complexes
 */
static PyObject *SCG_union(PyObject * self, PyObject * args)
{
	pyslsa_SCGObject *pyslsa_scg1, *pyslsa_scg2;

	if (!PyArg_ParseTuple(args, "OO", &pyslsa_scg1, &pyslsa_scg2))
		return NULL;

	SCG *scg1 = pyslsa_scg1->scg;
	SCG *scg2 = pyslsa_scg2->scg;

	struct simplex_hash_table *table = get_empty_hash_table_D();
	scg_list_union_hash(scg1, scg2, table);
	free_hash_table_D(table);

	return (PyObject *) pyslsa_scg2;
}

/*
 *  Compute the KL divergence between two python Simplicial Complexes 
 *  in dimension dim and with inverse temperature beta.
 *  NOT CUDA ACCELERATED
 */
static PyObject *KL(PyObject * self, PyObject * args)
{
	double beta, div;
	int dim;
	pyslsa_SCGObject *scg1, *scg2;

	if (!PyArg_ParseTuple(args, "OOid", &scg1, &scg2, &dim, &beta))
		return NULL;
	gsl_matrix *L1 = compute_simplicial_laplacian(scg1->scg, (size_t) dim);
	gsl_matrix *L2 = compute_simplicial_laplacian(scg2->scg, (size_t) dim);

	reconcile_laplacians(L1, L2, &L1, &L2);

	div = KL_divergence(L1, L2, beta);

	gsl_matrix_free(L1);
	gsl_matrix_free(L2);

	return Py_BuildValue("d", div);
}

/*
 *  Compute the KL divergence between two python Simplicial Complexes 
 *  in dimension dim and with inverse temperature beta.
 *  CUDA version of the above function
 */
static PyObject *cuKL(PyObject * self, PyObject * args)
{
	double beta, div;
	int dim;
	pyslsa_SCGObject *scg1, *scg2;

	if (!PyArg_ParseTuple(args, "OOid", &scg1, &scg2, &dim, &beta))
		return NULL;
	gsl_matrix *L1 = compute_simplicial_laplacian(scg1->scg, (size_t) dim);
	gsl_matrix *L2 = compute_simplicial_laplacian(scg2->scg, (size_t) dim);

	reconcile_laplacians(L1, L2, &L1, &L2);

	div = KL_divergence_cuda(L1, L2, beta);

	gsl_matrix_free(L1);
	gsl_matrix_free(L2);

	return Py_BuildValue("d", div);
}

/*
 *  Compute the Jensen-Shannon divergence between two python SCGs
 *  in dimension dim and with inverse temperature beta.
 *  NOT CUDA ACCELERATED
 */
static PyObject *JS(PyObject * self, PyObject * args)
{
	/* Compute the JS divergence between two simplices */
	double beta, div, div1, div2;
	int dim;
	pyslsa_SCGObject *scg1, *scg2;

	if (!PyArg_ParseTuple(args, "OOid", &scg1, &scg2, &dim, &beta))
		return NULL;
	gsl_matrix *L1 = compute_simplicial_laplacian(scg1->scg, (size_t) dim);
	gsl_matrix *L2 = compute_simplicial_laplacian(scg2->scg, (size_t) dim);

	reconcile_laplacians(L1, L2, &L1, &L2);
	gsl_matrix *M = gsl_matrix_alloc(L1->size1, L1->size2);
	gsl_matrix_memcpy(M, L1);

	/* Compute P + Q */
	gsl_matrix_add(M, L2);
	gsl_matrix_scale(M, 0.5);

	div1 = KL_divergence(L1, M, beta);
	div2 = KL_divergence(L2, M, beta);
	div = 0.5 * div1 + 0.5 * div2;

	gsl_matrix_free(L1);
	gsl_matrix_free(L2);
	gsl_matrix_free(M);

	return Py_BuildValue("d", div);
}

/*
 *  Compute the Jensen-Shannon divergence between two python SCGs
 *  in dimension dim and with inverse temperature beta.
 *  CUDA version of above function
 */
static PyObject *cuJS(PyObject * self, PyObject * args)
{
	/* Compute the JS divergence between two simplices */
	double beta, div, div1, div2;
	int dim;
	pyslsa_SCGObject *scg1, *scg2;

	if (!PyArg_ParseTuple(args, "OOid", &scg1, &scg2, &dim, &beta))
		return NULL;
	gsl_matrix *L1 = compute_simplicial_laplacian(scg1->scg, (size_t) dim);
	gsl_matrix *L2 = compute_simplicial_laplacian(scg2->scg, (size_t) dim);

	reconcile_laplacians(L1, L2, &L1, &L2);
	gsl_matrix *M = gsl_matrix_alloc(L1->size1, L1->size2);
	gsl_matrix_memcpy(M, L1);

	/* Compute P + Q */
	gsl_matrix_add(M, L2);
	gsl_matrix_scale(M, 0.5);

	div1 = KL_divergence_cuda(L1, M, beta);
	div2 = KL_divergence_cuda(L2, M, beta);
	div = 0.5 * div1 + 0.5 * div2;

	gsl_matrix_free(L1);
	gsl_matrix_free(L2);
	gsl_matrix_free(M);

	return Py_BuildValue("d", div);
}

/* 
 * Compute the pairwise JS divergence for a list of pairs of SCGs
 */
static PyObject *cuJS_pairs(PyObject * self, PyObject * args)
{
	double beta;
	double *divs;
	int dim, n_pairs, i;
	PyObject *scg_list, *pair, *ret_list;
	pyslsa_SCGObject *scg1, *scg2;

	if (!PyArg_ParseTuple(args, "Oid", &scg_list, &dim, &beta)) {
		return NULL;
	}

	n_pairs = PyList_Size(scg_list);
	gsl_matrix **mats = malloc(2 * n_pairs * sizeof(gsl_matrix *));
	//printf("Computing Laplacians...\n");
	for (i = 0; i < n_pairs; i++) {
		pair = PyList_GetItem(scg_list, i);
		scg1 = (pyslsa_SCGObject *) PyList_GetItem(pair, 0);
		scg2 = (pyslsa_SCGObject *) PyList_GetItem(pair, 1);
		//printf("Computing Laplacian %d:1...\n", i);
		//mats[2*i] = compute_simplicial_laplacian(scg1->scg, (size_t)dim);
		//printf("Computing Laplacian %d:2...\n", i);
		//mats[2*i+1] = compute_simplicial_laplacian(scg2->scg, (size_t)dim);
		mats[2 * i] = scg1->L1;
		mats[2 * i + 1] = scg2->L1;
	}
	//printf("Computing Divergences...\n");
	divs = cuda_par_JS(mats, n_pairs, beta);
	ret_list = PyList_New(n_pairs);
	for (i = 0; i < n_pairs; i++) {
		PyList_SetItem(ret_list, i, Py_BuildValue("d", divs[i]));
	}

	return ret_list;
}

/* 
 * Get the Laplacian spectra for the specified dimension for 
 * a list of SCGs
 */
static PyObject *PySLSA_get_laplacian_spectra(PyObject * self, PyObject * args)
{
	PyObject *scg_list;
	pyslsa_SCGObject *scg;
	gsl_vector **ress;
	int dim, n_scg, Lsize;
	int i, j;

	if (!PyArg_ParseTuple(args, "Oi", &scg_list, &dim)) {
		return NULL;
	}
	n_scg = (int)PyList_Size(scg_list);

	/* Allocate space for matrix list */
	gsl_matrix **mats = malloc(n_scg * sizeof(gsl_matrix *));

	/* Compute laplacians */
	for (i = 0; i < n_scg; i++) {
		scg = (pyslsa_SCGObject *) PyList_GetItem(scg_list, i);
		mats[i] = compute_simplicial_laplacian(scg->scg, (size_t) dim);
	}

	ress = cuda_batch_get_eigenvalues(mats, n_scg);

	/* Build Return Values */
	PyObject *eigvals_list = PyList_New(n_scg);
	for (i = 0; i < n_scg; i++) {
		Lsize = mats[i]->size1;
		PyObject *eigvals = PyList_New(Lsize);
		for (j = 0; j < Lsize; j++) {
			PyObject *val =
			    Py_BuildValue("d", gsl_vector_get(ress[i], j));
			PyList_SetItem(eigvals, j, val);
		}
		PyList_SetItem(eigvals_list, i, eigvals);
		gsl_matrix_free(mats[i]);
	}
	free(mats);

	return eigvals_list;
}

/*
 *  Define the functions available from the pycuslsa module
 */
static PyMethodDef pyslsa_funcs[] = {
	{"KL", (PyCFunction) KL, METH_VARARGS, NULL},
	{"JS", (PyCFunction) JS, METH_VARARGS, NULL},
	{"cuKL", (PyCFunction) cuKL, METH_VARARGS, NULL},
	{"cuJS", (PyCFunction) cuJS, METH_VARARGS, NULL},
	{"build_SCG", (PyCFunction) build_SCG, METH_VARARGS, NULL},
	{"union", (PyCFunction) SCG_union, METH_VARARGS, NULL},
	{"get_laplacian_spectra", (PyCFunction) PySLSA_get_laplacian_spectra,
	 METH_VARARGS, NULL},
	{"cuJS_pairs", (PyCFunction) cuJS_pairs, METH_VARARGS, NULL},
	{NULL}
};

/*
 *  Define the pycuslsa module
 */
static struct PyModuleDef pyslsa_module = {
	PyModuleDef_HEAD_INIT,
	"pycuslsa",
	pyslsa_docs,
	-1,
	pyslsa_funcs
};

/*
 *  Python initialization for the pycuslsa module
 */
PyMODINIT_FUNC PyInit_pycuslsa(void)
{
	PyObject *m;
	if (PyType_Ready(&pyslsa_SimplexType) < 0)
		return NULL;

	if (PyType_Ready(&pyslsa_SCGType) < 0)
		return NULL;

	m = PyModule_Create(&pyslsa_module);
	if (m == NULL)
		return NULL;

	Py_INCREF(&pyslsa_SimplexType);
	Py_INCREF(&pyslsa_SCGType);
	PyModule_AddObject(m, "Simplex", (PyObject *) & pyslsa_SimplexType);
	PyModule_AddObject(m, "SCG", (PyObject *) & pyslsa_SCGType);
	return m;
}
