/*
 * =====================================================================================
 *
 *       Filename:  pyslsa.c
 *
 *    Description:  PySLSA (Py salsa)  Python simplicial laplacian spectral 
 *                  analysis.
 *
 *        Version:  1.0
 *        Created:  10/20/2017 01:28:22 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Brad Theilman (BHT), bradtheilman@gmail.com
 *   Organization:  
 *
 * =====================================================================================
 */

#include <Python.h>

#include "simplex.h"

/* Simplex Object */
typedef struct {
    PyObject_HEAD
    struct Simplex * s;
} pyslsa_SimplexObject;

static PyObject * Simplex_new(PyTypeObject * type,
                              PyObject * args, PyObject * kwds)
{
    pyslsa_SimplexObject * self;
    self = (pyslsa_SimplexObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->s = create_empty_simplex();

    }
    return (PyObject *)self;
}

static void Simplex_dealloc(pyslsa_SimplexObject * self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void Simplex_free(pyslsa_SimplexObject * self)
{
    free_simplex(self->s);     
}

static PyObject * Simplex_add_vertex(pyslsa_SimplexObject * self, PyObject *args)
{
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) {
        return NULL;
    }
    add_vertex(self->s, i);
    Py_RETURN_NONE;
}

static PyObject * Simplex_dimension(pyslsa_SimplexObject * self)
{
    return Py_BuildValue("i", self->s->dim);
}

static PyMethodDef Simplex_methods[] = {
    {"add_vertex", (PyCFunction)Simplex_add_vertex, METH_VARARGS,
        "Add a vertex to the simplex"
    },
    {"get_dim", (PyCFunction)Simplex_dimension,
        METH_NOARGS, "Get Simplex Dimension"},
    {NULL}
};

static PyTypeObject pyslsa_SimplexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pyslsa.Simplex",
    sizeof(pyslsa_SimplexObject),
    0,
    (destructor)Simplex_dealloc,
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
    Simplex_methods, /* tp_methods */
    0, /* tp_members */
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Simplex_new,
    Simplex_free,      
};

/* SCG Object */
typedef struct {
    PyObject_HEAD
    SCG * scg;
} pyslsa_SCGObject;

static void SCG_dealloc(pyslsa_SimplexObject * self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void SCG_free(pyslsa_SimplexObject * self)
{
    free_SCG(self->s);     
}

static PyObject * SCG_new(PyTypeObject * type,
                              PyObject * args, PyObject * kwds)
{
    pyslsa_SCGObject * self;
    self = (pyslsa_SCGObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->scg = get_empty_SCG();

    }
    return (PyObject *)self;
}

static PyObject * PySCG_add_max_simplex(pyslsa_SCGObject * self,
                                      PyObject * args, PyObject *kwds)
{
    pyslsa_SimplexObject * maxsimp;
    if (!PyArg_ParseTuple(args, "O", &maxsimp))
        return NULL;
    scg_add_max_simplex(self->scg, maxsimp->s);
    Py_RETURN_NONE;
}

static PyMethodDef SCG_methods[] = {
    {"add_max_simplex", (PyCFunction)PySCG_add_max_simplex, METH_VARARGS,
        "Add a max simplex to the simplicial complex"
    },
    {NULL}
};

static PyTypeObject pyslsa_SCGType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pyslsa.SCG",
    sizeof(pyslsa_SCGType),
    0,
    (destructor)SCG_dealloc,
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
    SCG_methods, /* tp_methods */
    0, /* tp_members */
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    SCG_new,
    SCG_free,
};

/* MODULE METHODS AND DEFINITIONS */

static PyObject * helloworld(PyObject * self) 
{
    return Py_BuildValue("s", "Hello, Python!");
}

static char helloworld_docs[] = "Any Message Brad Theilman";

static PyMethodDef pyslsa_funcs[] = {
    {"helloworld", (PyCFunction)helloworld,
    METH_NOARGS, helloworld_docs},
    {NULL}
};

static struct PyModuleDef pyslsa_module = {
    PyModuleDef_HEAD_INIT,
    "pyslsa",
    helloworld_docs,
    -1,
    pyslsa_funcs
};


PyMODINIT_FUNC
PyInit_pyslsa(void)
{
    PyObject * m;
    if (PyType_Ready(&pyslsa_SimplexType) < 0)
        return NULL;

    if (PyType_Ready(&pyslsa_SCGType) < 0)
        return NULL;

    m = PyModule_Create(&pyslsa_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&pyslsa_SimplexType);
    Py_INCREF(&pyslsa_SCGType); 
    PyModule_AddObject(m, "Simplex", (PyObject *)&pyslsa_SimplexType);
    PyModule_AddObject(m, "SCG", (PyObject *)&pyslsa_SCGType);
    return m;
}
