import numpy as np 

def getXi(bin_mat, i):
	'''
	Returns ordered list of i-faces
	'''
	Xi = []
	# Compute the number of non-zero elements in each row
	rowSums = np.sum(bin_mat, 1)

	# Find all rows that correspond to i-faces
	iRowMask = (rowSums == i+1)
	iRows = bin_mat[iRowMask, :]

	(nRows, nCols) = np.shape(iRows)

	for rowIndx in range(nRows):
		row = np.squeeze(iRows[rowIndx, :])
		#find Nonzero Entries
		verts = np.arange(nCols)[row > 0]
		verts = np.sort(verts)
		Xi.append(verts)
	return np.sort(Xi)

def computeDiagonalEntry(Xi, Xil, Xip, alpha):
	'''
	Given list of i, (i-1), and (i+1) faces, 
	compute the alpha-th diagonal entry of ith laplacian
	'''

def computeOffDiagonalEntry(Xi, Xip, alpha, beta):
	'''
	Computes the alpha,beta entry in the ith laplacian
	'''

def computeSimplicialLaplacian(bin_mat, i):
	'''
	Return the ith simplicial laplacian matrix 
	computed from the bin_mat
	'''

