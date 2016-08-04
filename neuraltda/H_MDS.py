import numpy as np 
import scipy as sp 

def d_H(x1, x2):

	a = np.abs(x1-x2)
	b = np.abs(1.0 - x1*np.conj(x2))
	return 2*np.arctanh(a/b)

def ddH_dx1_1(x1, x2):

	x1_1 = np.real(x1)
	x1_2 = np.imaginary(x1)

	x2_1 = np.real(x2)
	x2_2 = np.imaginary(x2)

	v1 = x1_1 - x2_1
	v3 = x1_1*x2_1 + x1_2*x2_2 - 1
	v2 = x1_2 - x2_2 
	v4 = x1_1*x2_2 - x1_2*x2_1 
	tsq = (v1**2 + v2**2)/(v3**2 + v4**2)

	a = ((v1/(v1**2 + v2**2)) - (x2_1*v3 + x2_2*v4)/(v3**2 + v4**2))
	b = (2*np.sqrt(tsq)/(1-tsq))
	return b*a 

def ddH_dx1_2(x1, x2):

	x1_1 = np.real(x1)
	x1_2 = np.imaginary(x1)

	x2_1 = np.real(x2)
	x2_2 = np.imaginary(x2)

	v1 = x1_1 - x2_1
	v3 = x1_1*x2_1 + x1_2*x2_2 - 1
	v2 = x1_2 - x2_2 
	v4 = x1_1*x2_2 - x1_2*x2_1 
	tsq = (v1**2 + v2**2)/(v3**2 + v4**2)

	a = ((v2/(v1**2 + v2**2)) - (x1_1*v4 - x2_2*v3)/(v3**2 + v4**2))
	b = (2*np.sqrt(tsq)/(1-tsq))
	return b*a 

def mobius_xform(z, c, theta):

	a = theta*z + c 
	b = np.conj(c)*theta*z + 1
	return a/b

def E_x(D, w, d):

	diffmat = np.subtract(d, D)
	diffmat = np.square(diffmat)
	w = np.triu(w, k=1)
	E = np.einsum('ij,ij', w, diffmat)
	return E

def get_distances(X):

	dmat = np.zeros((len(X), len(X)))
	for i in range(len(X))
		for j in range(i):
			dmat[j, i] = d_H(X[i], X[j])
	return dmat + np.transpose(dmat)

def dE_dxa(D, w, X, alpha):

	d = get_distances(X)
	x_a = X[alpha]
	dd_vec = np.zeros(len(X))
	for j in range(len(X)):
		x_j = X[j]
		dd_real = ddH_dx1_1(x_a, x_j)
		dd_imag = ddH_dx1_2(x_a, x_j)
		dd = dd_real +1i*dd_imag 
		dd_vec[j] = dd 


	w = np.triu(w, k=1)
	w_j = w[alpha, :]
	diffmat = np.subtract(d[alpha, :] - D[alpha, :])
	dEdxa = 2*np.einsum('j,j,j', w_j, diffmat, dd_vec)
	return dEdxa

def HMDS_update(D, w, X, alpha):

	dE = dE_dxa(D,w,X,alpha)
	delta = eta*dE 
	X[alpha] = mobius_xform(X[alpha], delta, 1)




