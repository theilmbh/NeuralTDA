import numpy as np
from numpy import linalg as LA

class Simplex:
    def __init__(self, vertices, **kwargs):

        self.vertices=vertices
        self.faces=[]
        self.cofaces=[]
        self.dimension = len(self.vertices)-1
        self.absindex=None
        self.index=None
        self.label=None

        self._cfaces=[]
        self._children={}
        self._parent = None

class SimplicialComplex:

    def __init__(self, maximal_simplices):

        self.simplices = []
        self.edges = set()
        self.maximalSimplices = maximal_simplices
        self.dimension = max(map(len, self.maximalSimplices)) - 1
        self.nSimplexDict = {}
        for i in range(-1, self.dimension+1):
            self.nSimplexDict[i] = []

        self.createSimplex([]) # empty simplex
        self.setRoot(self.simplices[0])
        for ms in self.maximalSimplices:
            self.addMaximalSimplexOld(ms)

    def setRoot(self, simplex):
        self.root = simplex

    def getRoot(self):
        return self.root

    def createEdge(self, source, target):
        self.edges.add((source.absindex, target.absindex))

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

        if len(simplex.vertices) > 1:
            boundaryFaces = self._boundary(simplex)
            for face in boundaryFaces:
                self.createSimplex(face)
                newSimplex = self.simplices[-1]
                self.addFaceCoface(newSimplex, simplex)
                self.addBoundarySimplices(newSimplex)
        else:
            self.addFaceCoface(self.getRoot(), simplex)


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
        lowerSimps = self.nSimplexDict[n-2]
        boundary = []
        simpVect = np.zeros(len(lowerSimps))
        boundaryFaces = simplex.faces
        for index, face in enumerate(boundaryFaces):
            sgn = (-1)**(index+1)
            simpVect[face.index] = simpVect[face.index] + sgn
        return simpVect

    def getBoundaryMap(self, dimension):

        upperD = len(self.nSimplexDict[dimension])
        lowerD = len(self.nSimplexDict[dimension-1])

        boundaryMap = np.zeros((lowerD, upperD))
        for indx, sourceBasis in enumerate(self.nSimplexDict[dimension]):
            boundaryMap[:, indx] = self.getBoundaryChain(sourceBasis)
        return boundaryMap

    def getLaplacian(self, dimension):

        Di = self.getBoundaryMap(dimension)
        Di1 = self.getBoundaryMap(dimension+1)

        Lup = np.dot(Di1, np.transpose(Di1))
        Ldown = np.dot(np.transpose(Di), Di)

        return np.add(Lup, Ldown)

    def getSpectrum(self, dimension):

        L = self.getLaplacian(dimension)
        w, v = LA.eig(L)
        return w

def test_binmat(Ncells, Nwin):

    randBinMat = np.random.random((Ncells, Nwin))

    maxSimps = []
    for cell in range(Ncells):

        activeInds = np.arange(Nwin)[randBinMat[cell, :]>0.8]
        maxSimps.append(activeInds)

    return SimplicialComplex(maxSimps)