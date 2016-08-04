import numpy as np 
import scipy as sp 
from scipy.misc import derivative

def d_H(x1, x2):

	a = np.abs(x1-x2)
	b = np.abs(1.0 - x1*np.conj(x2))
	return 2*np.arctanh(a/b)

def ddH_dx1_1(x1, x2):

	x1_1 = np.real(x1)
	x1_2 = np.imag(x1)

	x2_1 = np.real(x2)
	x2_2 = np.imag(x2)

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
	x1_2 = np.imag(x1)

	x2_1 = np.real(x2)
	x2_2 = np.imag(x2)

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

def E_x(X, Y, D, w):

	d = get_distances(X, Y)
	diffmat = np.subtract(d, D)
	diffmat = np.square(diffmat)
	
	E = 0.5*np.einsum('ij,ij', w, diffmat)
	return E

def get_distances(X, Y):

	dmat = np.zeros((len(X), len(X)))
	for i in range(len(X)):
		for k in range(i):
			dmat[k, i] = d_H(X[i]+1j*Y[i], X[k]+1j*Y[k])
	return dmat + np.transpose(dmat)

def dE_dxa(D, w, X, Y, alpha, q):

	d = get_distances(X, Y)
	x_a_1 = X[alpha]
	x_a_2 = Y[alpha]
	x_a = x_a_1 + 1j*x_a_2
	dd_vec = np.zeros(len(X))
	if q==1:
		ddH = lambda x, y: ddH_dx1_1(x, y)
	elif q==2:
		ddH = lambda x, y: ddH_dx1_2(x, y)
	rng = range(len(X))
	
	for j in rng:
		x_j = X[j] + 1j*Y[j]
		dd = ddH(x_a, x_j)
		dd_vec[j] = np.real(dd)
		dd_vec[alpha]=0.0

	w_j = w[alpha, :]
	diffmat = np.subtract(d[alpha, :],  D[alpha, :])
	dEdxa = np.einsum('j,j,j', w_j, diffmat, dd_vec)
	return dEdxa

def dE_dxa_q(x, D, w, X, Y, alpha, q):

	a = 0
	if q==1:
		X[alpha]=x
		a = dE_dxa(D, w, X, Y, alpha, q)
	elif q==2:
		Y[alpha] = x
		a = dE_dxa(D,w, X, Y, alpha, q)
	return a

def HMDS_update(D, w, X, Y, alpha, eta):

	dE_1 = dE_dxa(D,w,X,Y,alpha, 1)
	dE_2 = dE_dxa(D, w, X, Y, alpha, 2)
	ddE_dxa_1 = lambda x: dE_dxa_q(x, D, w, X, Y, alpha, 1)
	ddE_dxa_2 = lambda y: dE_dxa_q(y, D, w, X, Y, alpha, 2)
	ddEdxa1 = derivative(ddE_dxa_1, X[alpha], dx=1e-4)
	ddEdxa2 = derivative(ddE_dxa_1, Y[alpha], dx=1e-4)

	delta_r = dE_1 / np.abs(ddEdxa1)
	delta_i = dE_2 / np.abs(ddEdxa2)

	#delta_r = dE_1 / 1.0
	#delta_i = dE_2 / 1.0

	delta = -1.0*eta*(delta_r +1j*delta_i) 
	new = mobius_xform(delta, X[alpha]+1j*Y[alpha], 1)
	return new





