import numpy as np 
from itertools import combinations

def getXi(bin_mat, i):
	'''
	Returns ordered list of i-faces
	'''
	Xi = set()
	# Compute the number of non-zero elements in each row
	rowSums = np.sum(bin_mat, 1)

	# Find all rows that correspond to i-faces
	iRowMask = (rowSums >= i+1)
	iRows = bin_mat[iRowMask, :]

	(nRows, nCols) = np.shape(iRows)

	for rowIndx in range(nRows):
		row = np.squeeze(iRows[rowIndx, :])
		#find Nonzero Entries
		verts = np.arange(nCols)[row > 0]
		verts = np.sort(verts)
		verts = (list(s) for s in combinations(verts, i+1))
		Xi.add(verts)
	return np.sort(list(Xi))

def computeDiagonalEntry(Xi, Xil, Xip, alpha):
	'''
	Given list of i, (i-1), and (i+1) faces, 
	compute the alpha-th diagonal entry of ith laplacian
	'''

def computeOffDiagonalEntry(Xi, Xip, alpha, beta, i):
	'''
	Computes the alpha,beta entry in the ith laplacian
	'''
	(NFaces, ip1) = np.shape(Xi)
	XiAlpha = Xi[alpha, :]
	XiBeta = Xi[beta, :]
	# compute union
	XaUXb = np.union1d(XiAlpha, XiBeta)
	if not np.size(XaUXb) == i+1:
		return 0
	elementMask = ((Xi-XaUXb)==0).sum(1)==ip1
	if elementMask.any():
		return 0

	XaXbdiff = XiAlpha-XiBeta
	XaXbDiffIndx = np.arange(ip1)[not XaXbdiff==0]
	return (-1)**(XaXbDiffIndx)

def computeLUpOffDiagonal(Xa, Xb, XiU):

	ip1 = np.size(Xa)
	XUnion = np.union1d(Xa, Xb)
	if not np.size(XUnion) == ip1+1:
		return 0
	elementMask = ((XiU-XUnion)==0).sum(1)==ip1+1
	if elementMask.any():
		XaXbDiff = Xa - Xb
		nDiff = (not XaXbDiff==0).sum()
		return (-1)**nDiff
	else:
		return 0


def computeUpperDegree(Xa, XiU):

	(NRows, ip11) = np.shape(XiU)
	deg = 0
	for row in range(NRows):
		if (set(Xa) < set(XiU[row, :])):
			deg = deg+1
	return deg

def computeLUp(Xi, XiU):

	(N, ip1) = np.shape(Xi)
	LUp = np.zeros((N, N))
	for alpha in range(N):
		for beta in range(N):
			Xa = Xi[alpha, :]
			Xb = Xi[beta, :]
			if alpha==beta:
				LUp[alpha, beta] = computeUpperDegree(Xa, XiU)
			LUp[alpha, beta] = computeLUpOffDiagonal(Xa, Xb, XiU)
	return LUp

def computeLDown(Xi, XiD):

	return 0


def computeSimplicialLaplacian(bin_mat, i):
	'''
	Return the ith simplicial laplacian matrix 
	computed from the bin_mat
	'''

