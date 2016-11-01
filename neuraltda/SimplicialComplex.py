import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from scipy import linalg as LAS 

class Simplex:
    def __init__(self, vertices, **kwargs):

        self.vertices=vertices
        self.faces=[]
        self.cofaces=[]
        self.dimension = len(self.vertices)-1
        self.absindex=None
        self.index=None
        self.label=None
        self.boundaryChain = None

        self._cfaces=[]
        self._children={}
        self._parent = None

class SimplicialComplex:

    def __init__(self, maximal_simplices, collapseVertices=False, name=''):

        assert len(maximal_simplices) > 0, 'Empty Simplex!'
        self.simplices = []
        self.edges = set()
        self.maximalSimplices = maximal_simplices
        self.dimension = max(map(len, self.maximalSimplices)) - 1
        self.nSimplexDict = {}
        self.LUpDict = {}
        self.LDownDict = {}
        self.LaplacianDict = {}
        self.spectrum = {}
        self.entropy = {}
        self.name = name
        for i in range(-1, self.dimension+1):
            self.nSimplexDict[i] = []
            self.LUpDict[i] = []
            self.LDownDict[i] = []
            self.LaplacianDict[i] = []
            self.spectrum[i] = []
            self.entropy[i] = []
            self.densityMatrix[i] = []

        self.createSimplex([]) # empty simplex
        self.setRoot(self.simplices[0])

        if collapseVertices:
            maxVertex = max(map(max, self.maximalSimplices))
            self.createVertices(maxVertex)
            for ms in self.maximalSimplices:
                self.addMaximalSimplex(ms)

        else:
            print('Building simplex..')
            for ms in tqdm(self.maximalSimplices):
                self.addMaximalSimplexOld(ms)

    def setRoot(self, simplex):
        self.root = simplex

    def getRoot(self):
        return self.root

    def createEdge(self, source, target):
        self.edges.add((source.absindex, target.absindex))

    def createVertices(self, maxvertex):
        for vert in range(maxvertex):
            self.createSimplex([vert])
            newSimplex = self.simplices[-1]
            self.addFaceCoface(self.getRoot(), newSimplex)

    def createSimplex(self, vertices):

        newSimplex = Simplex(vertices)
        self.simplices.append(newSimplex)
        self.nSimplexDict[newSimplex.dimension].append(self.simplices[-1])
        index = len(self.nSimplexDict[newSimplex.dimension])-1
        newSimplex.index = index
        newSimplex.absindex = len(self.simplices)-1

    def createChild(self, simplex, vertex):

        if vertex in simplex._cfaces:
            return
        simplex._cfaces.append(vertex)
        childVertices = [v for v in simplex.vertices]
        childVertices.append(vertex)
        self.createSimplex(childVertices)
        childSimplex = self.simplices[-1]
        childSimplex._parent=simplex
        simplex._children[vertex]=childSimplex
        self.createEdge(simplex, childSimplex)
        simplex.cofaces.append(childSimplex)
        childSimplex.faces.append(simplex)

    def addMaximalSimplexOld(self, vertices, simplex=None, first=True):
        #We start at the root.
        if first:
            simplex = self.getRoot()

        if len(vertices)>=1: #Condition to terminate recursion.
            for index in range(len(vertices)):
                vertex = vertices[index]
                self.createChild(simplex, vertex)
                self.addMaximalSimplexOld(simplex=simplex._children[vertex],
                                       vertices=vertices[index+1:],
                                       first=False)

    def addMaximalSimplex(self, vertices):

        self.createSimplex(vertices)
        newMaxSimplex = self.simplices[-1]
        self.addBoundarySimplices(newMaxSimplex)

    def addBoundarySimplices(self, simplex):

        if len(simplex.vertices) > 2:
            boundaryFaces = self._boundary(simplex)
            for face in boundaryFaces:
                self.createSimplex(face)
                newSimplex = self.simplices[-1]
                self.addFaceCoface(newSimplex, simplex)
                self.addBoundarySimplices(newSimplex)
        if len(simplex.vertices) == 2:
            v1 = self.getSimplex([simplex.vertices[0]])
            v2 = self.getSimplex([simplex.vertices[1]])
            self.addFaceCoface(v1, simplex)
            self.addFaceCoface(v2, simplex)

    def getSimplex(self, address, first=True, simplex=None):
        if first:
            simplex = self.getRoot()
        if len(address)==0:
            return simplex
        if address[0] not in simplex._cfaces:
            return None
        return self.getSimplex(address=address[1:], first=False,
                               simplex=simplex._children[address[0]])

    def _boundary(self, simplex):

        vertices = simplex.vertices
        n = len(vertices)
        boundary = []
        for index in range(n):
            boundary.append(vertices[:index] + vertices[index+1:])
        return boundary

    def addFaceCoface(self, face, coface):

        if coface._parent != face:
            self.createEdge(face, coface)
            face.cofaces.append(coface)
            coface.faces.append(face)

    def updateAdjacencySimplex(self, simplex):
        boundaryFaces = self._boundary(simplex)
        boundaryFaces = map(self.getSimplex, boundaryFaces)
        for face in boundaryFaces:
            self.addFaceCoface(face, simplex)

    def updateAdjacency(self):
        for i in range(self.dimension+1):
            for simplex in self.nSimplexDict[i]:
                self.updateAdjacencySimplex(simplex)

    def getBoundaryChain(self, simplex):

        vertices = simplex.vertices
        n = len(vertices)
        lowerSimps = self.nSimplexDict[simplex.dimension-1]
        boundary = []
        simpVect = np.zeros(len(lowerSimps))
        boundaryFaces = simplex.faces
        for index, face in enumerate(boundaryFaces):
            pos = set(simplex.vertices) - set(face.vertices)
            simpVect[face.index] = (-1)**(simplex.vertices.index(list(pos)[0]))
        simplex.boundaryChain = simpVect
        return simpVect

    def getBoundaryMap(self, dimension):

        upperD = len(self.nSimplexDict[dimension])
        lowerD = len(self.nSimplexDict[dimension-1])

        boundaryMap = np.zeros((lowerD, upperD))
        for indx, sourceSimplex in enumerate(self.nSimplexDict[dimension]):
            boundaryMap[:, indx] = self.getBoundaryChain(sourceSimplex)
        return boundaryMap

    def getLaplacian(self, dimension):

        if self.LaplacianDict[dimension]:
            return self.LaplacianDict[dimension]
        else:
            Di = self.getBoundaryMap(dimension)
            Di1 = self.getBoundaryMap(dimension+1)

            Lup = np.dot(Di1, np.transpose(Di1))
            Ldown = np.dot(np.transpose(Di), Di)
            L = np.add(Lup, Ldown)
            self.LaplacianDict[dimension] = L
            return L

    def getSpectrum(self, dimension):

        if self.spectrum[dimension]:
            return self.spectrum[dimension]
        else:
            L = self.getLaplacian(dimension)
            w, v = LA.eig(L)
            self.spectrum[dimension] = w
            return w

    def computeSpectralEntropy(self, dimension, beta):

        if self.entropy[dimension]:
            return self.entropy[dimension]
        else:
            spec = self.getSpectrum(dimension)

            gibbsdist = np.exp(beta*spec)
            gibbsdist = gibbsdist/sum(gibbsdist)

            entropy = -1.0*sum(np.multiply(gibbsdist, np.log(gibbsdist)/np.log(2)))
            self.entropy[dimension] = entropy
            return entropy

    def computeDensityMatrix(self, dimension, beta):

        if self.densityMatrix[dimension]:
            return self.densityMatrix[dimension]
        else:
            L = self.getLaplacian(dimension)
            rho = np.exp(beta*L)
            rho = rho / np.trace(rho)
            self.densityMatrix[dimension] = rho
            return rho

    def computeKLDivergence(self, dimension, beta, sigmaComplex):

        rho = self.computeDensityMatrix(dimension, beta)
        sigma = sigmaComplex.computeDensityMatrix(dimension, beta)



    def getUpperCommonSimplex(self, s1, s2):

        s1c = s1.cofaces
        s2c = s2.cofaces
        return list(set(s1c) & set(s2c))

    def getLowerCommonSimplex(self, s1, s2):

        s1f = s1.faces
        s2f = s2.faces
        return list(set(s1f) & set(s2f))

    def computeUpperLaplacianDirect(self, dimension):

        if self.LUpDict[dimension]:
            return self.LUpDict[dimension]
        else:
            dsimplices = self.nSimplexDict[dimension]
            nSimplices = len(dsimplices)
            LUp = np.zeros((nSimplices, nSimplices))
            for sind1i in range(nSimplices):
                for sind2i in range(sind1i, nSimplices, 1):
                    s1 = dsimplices[sind1i]
                    s2 = dsimplices[sind2i]
                    sind1 = s1.index
                    sind2 = s2.index
                    if sind1 == sind2:
                        degU = len(s1.cofaces)
                        LUp[sind1, sind2] = degU
                    else:
                        UCS = self.getUpperCommonSimplex(s1, s2)
                        if UCS:
                            # Find orientation rel to upper common simplex
                            vertdiff = [abs(v1-v2) for v1, v2 in zip(s1.vertices, s2.vertices)]
                            sgn = (-1)**(sum(vertdiff))
                            LUp[sind1, sind2] = sgn
                            LUp[sind2, sind1] = sgn
            self.LUpDict[dimension] = LUp
            return LUp

    def computeLowerLaplacianDirect(self, dimension):

        if self.LDownDict[dimension]:
            return self.LDownDict[dimension]
        else:

            dsimplices = self.nSimplexDict[dimension]
            nSimplices = len(dsimplices)
            LDown = np.zeros((nSimplices, nSimplices))
            for sind1i in range(nSimplices):
                for sind2i in range(sind1i, nSimplices, 1):
                    s1 = dsimplices[sind1i]
                    s2 = dsimplices[sind2i]
                    sind1 = s1.index
                    sind2 = s2.index
                    if sind1 == sind2:
                        degL = len(s1.faces)
                        LDown[sind1, sind2] = degL
                    else:
                        LCS = self.getLowerCommonSimplex(s1, s2)
                        if LCS:
                            vect1 = self.getBoundaryChain(s1)
                            vect2 = self.getBoundaryChain(s2)
                            sgn = vect1[LCS[0].index]*vect2[LCS[0].index]
                            LDown[sind1, sind2] = sgn
                            LDown[sind2, sind1] = sgn
            self.LDownDict[dimension] = LDown
            return LDown

def test_binmat(Ncells, Nwin):

    randBinMat = np.random.random((Ncells, Nwin))

    maxSimps = []
    for cell in range(Ncells):

        activeInds = np.arange(Nwin)[randBinMat[cell, :]>0.8]
        maxSimps.append(activeInds)

    return SimplicialComplex(maxSimps)

def binnedtobinary(popvec, thresh):
    '''
    Takes a popvec array from a binned data file and converts to a binary matrix according to thresh

    Parameters
    ----------
    popvec : array 
        An NCells by Nwindow array containing firing rates in that window.  
    Thresh : float 
        Multiple of average firing rate to use for thresholding
    '''

    popvec = np.array(popvec)
    Ncells, Nwin = np.shape(popvec)
    means = popvec.sum(1)/Nwin
    means = np.tile(means, (Nwin, 1)).T
    meanthr = thresh*means
    
    activeUnits = np.greater(popvec, meanthr).astype(int)
    return activeUnits
    
def BinaryToMaxSimplex(binMat):
    ''' 
    Takes a binary matrix and computes maximal simplices according to CI 2008 

    Parameters
    ----------
    binMat : numpy array 
        An Ncells x Nwindows array 
    '''

    Ncells, Nwin = np.shape(binMat)
    MaxSimps = []
    for cell in range(Ncells):
        if binMat[cell, :].any():
            verts = np.arange(Nwin)[binMat[cell, :] == 1]
            verts = np.sort(verts)
            MaxSimps.append(list(verts))
    return MaxSimps
        
def ShuffleBinary(binMat):
    retMat = np.array(binMat)
    Ncells, Nwin = np.shape(binMat)
    for cell in range(Ncells):
        np.random.shuffle(retMat[cell, :])
    return retMat

def computeRD(rho, sigma, q):
    
    mat = np.dot(LAS.fractional_matrix_power(rho, q), LAS.fractional_matrix_power(sigma, 1.0-q))
    return np.log(np.trace(mat))/(np.log(2)*(q-1))

def computeRenyiEntropy(rho, q):

    return np.log(np.trace(LAS.fractional_matrix_power(rho, q)))/(np.log(2)*(q-1))

def computeJSDivergence(rho, sigma, q):

    rho = self.densityMatrix[dimension]
    mix = (rho+sigma)/2.0
    J = computeRenyiEntropy(mix, q) - (0.5)*(computeRenyiEntropy(rho, q) + computeRenyiEntropy(sigma, q))