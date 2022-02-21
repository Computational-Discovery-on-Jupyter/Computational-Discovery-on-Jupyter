import numpy as np
import math
from PIL import Image
import matplotlib as mpl
from matplotlib import cm
from itertools import accumulate

class DensityPlot:
    def __init__(self, bounds, nrow, ncol):
        self._bounds = bounds #bounds = [left, right, bottom, top]
        self._nrow = nrow
        self._ncol = ncol
        self._dx = (bounds[1] - bounds[0])/ncol
        self._dy = (bounds[3] - bounds[2])/nrow
        self._matrix = np.zeros((nrow, ncol), dtype = int)
        self._rescaleMatrix = None
        self._rescaleMatrixColor = None
        self._colorscale = None

    def getMatrix(self):
        return(self._matrix)

    def _computeLocation(self, root):
        i = self._nrow - math.floor((root.imag - self._bounds[2])/self._dy)
        j = math.floor((root.real - self._bounds[0])/self._dx)
        return(i, j)

    def addPoints(self, roots):
        self._rescaleMatrix = None
        self._rescaleMatrixColor = None
        for root in roots:
            (i, j) = self._computeLocation(root)
            if i > -1 and i < self._nrow and j > -1 and j < self._ncol:
                self._matrix[i, j] += 1

    def addMultiplePoints(self, roots, nrepeats ):
        self._rescaleMatrix = None
        self._rescaleMatrixColor = None
        for root in roots:
            (i, j) = self._computeLocation(root)
            if i > -1 and i < self._nrow and j > -1 and j < self._ncol:
                self._matrix[i, j] += nrepeats

    def getFrequencies(self):
        u, c = np.unique(np.array(self._matrix), return_counts=True)
        # np.stack([u, c]).T
        return(u, c)

    def _rescaleMap(self):
        u, c = self.getFrequencies()
        if self._colorscale == "log":
            maxDensity = math.log(u[-1])
            minDensity = math.log(u[1])
            remap_func = lambda x: (1 + (math.log(x) - minDensity)/(maxDensity - minDensity)*256)/257
            self._rescaleMatrixColor = "log"
        elif self._colorscale == "linear":
            maxDensity = u[-1]
            minDensity = u[1]
            remap_func = lambda x: (1 + (x - minDensity)/(maxDensity - minDensity)*256)/257
            self._rescaleMatrixColor = "linear"
        else:
            cuf = [0] + list(accumulate(c[1:-1]))
            map_dict = dict(zip(list(u[1:]), cuf))
            remap_func = lambda x: (1 + map_dict[x]/cuf[-1]*256)/257
            self._rescaleMatrixColor = "cumulative"
        self._rescaleMatrix = np.zeros((self._nrow, self._ncol))
        for i in range(self._nrow):
            for j in range(self._ncol):
                if self._matrix[i, j] != 0:
                    self._rescaleMatrix[i, j] = remap_func(self._matrix[i, j])

    def getRescaleMatrix(self):
        return(self._rescaleMatrix)

    def makeDensityPlot(self, cmap_name, **kwargs):
        filename = kwargs.get('filename', None)
        bgcolor = kwargs.get('bgcolor', [0, 0, 0, 1])
        colorscale = kwargs.get('colorscale', 'cumulative')
        if colorscale not in ['log', 'linear', 'cumulative']:
            raise ValueError('Unexpected value imported; expected the string "log", "linear", or "cumulative".')
        else:
            self._colorscale = colorscale            
        if self._rescaleMatrix is None or self._colorscale != self._rescaleMatrixColor:
            self._rescaleMap()
        getcmap = cm.get_cmap(cmap_name)
        bgarray = np.array(bgcolor)
        cmap = np.vstack((bgarray, getcmap(np.arange(256))))
        cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
        img = Image.fromarray(np.uint8(cmap(self._rescaleMatrix)*255))
        display(img)
        if filename is not None:
            img.save(filename)
