import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

class Bohemian:
    """
    Python class to create different families of Bohemian matrices. This class includes a method to calculate the characteristic polynomial, determinant, and eigenvalues for each respective Bohemian family.
    """
    def __init__(self, n, U=None):
        """
        Initialization of Bohemian object.
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) entries of the matrix. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n*n
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for i in range(n):
                for j in range(n):
                    self._matrix[i, j] = U[k]
                    k += 1
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) entries of the matrix (length of matrix needs to match the indicated size of initialized Bohemian matrix object)
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for i in range(self._size):
            for j in range(self._size):
                self._matrix[i, j] = U[k]
                k += 1

    def resizeMatrix(self, n):
        """
        Function to resize the Bohemian matrix object
        Input:
        n: (int) size of matrix (assume that Bohemian matrix object is a square matrix)
        """
        self._size = n
        self._matrix = np.zeros((n, n), dtype=type(self._matrix[0,0]))

    def getMatrix(self):
        """
        Function that returns the Bohemian matrix
        Output:
        (numpy.array) Bohemian matrix
        """
        return(self._matrix)

    def getSize(self):
        """
        Function that returns the size of the Bohemian matrix
        Output:
        (int) size of matrix
        """
        return(self._size)

    def getNumberOfMatrixEntries(self):
        """
        Function that returns the number of entries in the matrix dependent on the type of Bohemian family indicated when object is initialized
        Output:
        (int) number of matrix entries
        """
        return(self._numberOfMatrixEntries)

    def characteristicPolynomial(self):
        """
        Function that calculates the characteristic polynomial of the Bohemian matrix
        Output:
        (numpy array) Coefficients of the characteristic polynomial of the Bohemian matrix (in order of descending order of power)
        """
        # Use the method of Fadeev-Leverrier to compute
        # the characteristic polynomial.  Uses integer arithmetic
        # although numpy insists on floats sometimes.
        # This algorithm is inefficient with O(n^4) operations in general
        # and with the possibility of very large numbers in it anyway.
        # But it will work over anything that can multiply and integer divide.
        M = np.zeros((self._size, self._size), dtype=type(self._matrix[0,0]))
        # Rely on flint arithmetic
        c = np.zeros(self._size + 1, dtype=type(self._matrix[0,0]))
        c[self._size] = 1
        B = np.zeros((self._size, self._size), dtype=type(self._matrix[0,0])) # A.M = 0 b/c M=0
        for k in range(self._size):
            M = B + c[self._size-k]*np.identity(self._size)
            for i in range(self._size):
                for jay in range(self._size):
                    B[i,jay] = 0
                    for ell in range(self._size):
                        B[i,jay] += self._matrix[i,ell]*M[ell,jay]
            # Hard-coding matrix multiply
            c[self._size-k-1] = 0
            for i in range(self._size):
                c[self._size-k-1] += B[i,i]
            c[self._size-k-1] = -c[self._size-k-1]/(k+1) # Division must be exact; result might be "flint"
        return c

    def determinant(self):
        """
        Function that calculates the determinant of the Bohemian matrix
        Output:
        (int) Determinant of the Bohemian matrix
        """
        c = self.characteristicPolynomial()
        return (-1)**self._size*c[0]

    def eig(self):
        """
        Function that calculates the eigenvalues of the Bohemian matrix
        Output:
        (numpy.array) All eigenvalues of the Bohemian matrix
        """
        eigval, _ = la.eig(self._matrix)
        return(eigval)

    def plotEig(self):
        """
        Function that plots the eigenvalues of the Bohemian matrix.
        Output:
        (matplotlib.plot) Plot of eigenvalues (with default colours/parameters)
        """
        e = self.eig()
        x = e.real
        y = e.imag
        plt.scatter(x, y)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()

class Symmetric(Bohemian):
    """
    Inherited class to create symmetric Bohemian matrix
    """
    def __init__(self, n, U=None):
        """
        Initialization of Symmetric Bohemian Matrix Object
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) upper triangular entries of the matrix. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n*(n+1)//2
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for i in range(n):
                for j in range(i, n):
                    self._matrix[i, j] = U[k]
                    k += 1
            for i in range(n):
                for j in range(i+1, n):
                    self._matrix[j, i] = self._matrix[i, j]
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) upper triangular entries of the matrix (length of matrix needs to match the indicated size of initialized Bohemian matrix object)
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for i in range(self._size):
            for j in range(i, self._size):
                self._matrix[i, j] = U[k]
                k += 1
        for i in range(self._size):
            for j in range(i+1, self._size):
                self._matrix[j, i] = self._matrix[i, j]

class SkewSymmetric(Bohemian):
    """
    Inherited class to create skew symmetric Bohemian matrix
    """
    def __init__(self, n, U=None):
        """
        Initialization of Skew Symmetric Bohemian Matrix Object
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) upper triangular entries of the matrix. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n*(n-1)//2
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    self._matrix[i, j] = U[k]
                    k += 1
            for i in range(n):
                for j in range(i+1, n):
                    self._matrix[j, i] = -self._matrix[i, j]
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) upper triangular entries of the matrix (length of matrix needs to match the indicated size of initialized Bohemian matrix object)
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for i in range(self._size):
            for j in range(i+1, self._size):
                self._matrix[i, j] = U[k]
                k += 1
        for i in range(self._size):
            for j in range(i+1, self._size):
                self._matrix[j, i] = -self._matrix[i, j]

# Add a class of skew-symmetric tridiagonal matrices.
class SkewSymmetricTridiagonal(Bohemian):
    """
    Inherited class to create skew symmetric tridiagonal Bohemian matrix
    """
    def __init__(self, n, U=None):
        """
        Initialization of Skew Symmetric Triadiagonal Bohemian Matrix Object
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) entries of the matrix, length of entries need to match the number of matrix entries. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n-1
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for i in range(n-1):
                self._matrix[i, i+1] = U[k]
                k += 1
            for i in range(n-1):
                self._matrix[i+1, i] = -self._matrix[i, i+1]
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) entries of the matrix, length of entries need to match the number of matrix entries
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for i in range(self._size-1):
            self._matrix[i, i+1] = U[k]
            k += 1
        for i in range(self._size-1):
            self._matrix[i+1, i] = -self._matrix[i, i+1]

    def characteristicPolynomial(self):
        """
        Function to calculate the characteristic polynomial of Skew Symmetric Triangular Bohemian Object
        Output:
        (numpy.array) Array of coefficients of the characteristic polynomial (in descending order of power)
        """
        # This routine is special for skew-symmetric
        # tridiagonal matrices.  There is a fast recurrence
        # for computing characteristic polynomials for this class.
        # See R. M. Corless, https://doi.org/10.5206/mt.v1i2.14360
        # We here use numpy.polynomial.Polynomial arithmetic
        # to simulate symbolic computation.  What is returned
        # is an array of coefficients (which can later be cast
        # as a tuple or Polynomial, as desired)
        c = np.zeros(self._size + 1, dtype=type(self._matrix[0,0]))
        c[self._size] = 1
        p0 = np.polynomial.Polynomial([1])
        p1 = np.polynomial.Polynomial([0,1])
        mu = np.polynomial.Polynomial([0,1])
        for k in range(self._size-1):
            p = mu*p1 + self._matrix[k,k+1]**2*p0
            p0 = p1
            p1 = p
        # Leading coefficient is already 1.
        for k in range(self._size):
            c[k] = p.coef[k]
        return c


# Add a class of upper Hessenberg Toeplitz Zero Diagonal matrices with -1 on the subdiagonal
# This code has been minimally tested, but has passed those tests.
class UHTZD(Bohemian):
    """
    Inherited class to create upper Hessenberg Toeplitz Zero Diagonal Bohemian matrix with -1 on the subdiagonal
    """
    def __init__(self, n, U=None):
        """
        Initialization of upper Hessenberg Toeplitz Zero Diagonal Bohemian matrix with -1 on the subdiagonal object
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) entries of the matrix, length of entries need to match the number of matrix entries. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n-1
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for j in range(1,n):
                self._matrix[0, j] = U[k]
                k += 1
                for i in range(1,n-j):
                    self._matrix[i,j+i] = self._matrix[0,j]
            for i in range(n-1):
                self._matrix[i+1, i] = -1
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) entries of the matrix, length of entries need to match the number of matrix entries
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for j in range(1,self._size):
            self._matrix[0, j] = U[k]
            k += 1
            for i in range(1,self._size-j):
                self._matrix[i,j+i] = self._matrix[0,j]
        for i in range(self._size-1):
            self._matrix[i+1, i] = -1

    def characteristicPolynomial(self):
        """
        Function to calculate the characteristic polynomial of upper Hessenberg Toeplitz Zero Diagonal Bohemian matrix with -1 on the subdiagonal
        Output:
        (numpy.array) Array of coefficients of the characteristic polynomial (in descending order of power)
        """
        # This routine is special for upper Hessenberg Toeplitz
        # zero diagonal matrices.  There is a fast recurrence
        # for computing characteristic polynomials for this class.
        # P(n) = mu*P(n - 1) + add((-1)^k*t[k]*P(n - k), k = 1 .. n)
        # See https://doi.org/10.1016/j.laa.2020.03.037
        # We here use numpy.polynomial.Polynomial arithmetic
        # to simulate symbolic computation.  What is returned
        # is an array of coefficients (which can later be cast
        # as a tuple or Polynomial, as desired)
        c = np.zeros(self._size + 1, dtype=type(self._matrix[0,0]))
        c[self._size] = 1
        plist = []
        plist.append( np.polynomial.Polynomial([1]) )
        plist.append( np.polynomial.Polynomial([0,1]) )
        mu = np.polynomial.Polynomial([0,1])
        for n in range(2,self._size+1):
            s = 0
            for j in range(1,n):
                 s += (-1)**j*self._matrix[0,j]*plist[n-j-1]
            p = mu*plist[n-1] - s
            plist.append( p )
        # So many opportunities for sign errors and off-by-one errors :(
        for k in range(self._size):
            c[k] = plist[self._size].coef[k]
        return c


# Add a class of Unit upper Hessenberg matrices with 1 on the subdiagonal
class UnitUpperHessenberg(Bohemian):
    """
    Inherited class to create unit upper Hessenberg Toeplitz Bohemian matrix
    """
    def __init__(self, n, U=None):
        """
        Initialization ofunit upper Hessenberg Toeplitz Bohemian matrix
        Input:
        n: (int) size of the matrix (Bohemian matrix is assumed to be square)
        U: (list, optional) entries of the matrix, length of entries need to match the number of matrix entries. Default will result in a zero matrix of size n x n
        """
        self._size = n
        self._numberOfMatrixEntries = n*(n+1)//2
        if U is not None:
            self._matrix = np.zeros((n, n), dtype=type(U[0]))
            if self._numberOfMatrixEntries != len(U):
                raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
            k = 0
            for i in range(n):
                for j in range(i,n):
                    self._matrix[i, j] = U[k]
                    k += 1
            for i in range(n-1):
                self._matrix[i+1, i] = 1
        else:
            self._matrix = np.zeros((n, n), dtype=int)

    def makeMatrix(self, U):
        """
        Function to input entries for Bohemian matrix object.
        Input:
        U: (list) entries of the matrix, length of entries need to match the number of matrix entries
        """
        if self._numberOfMatrixEntries != len(U):
            raise ValueError('Expected list of length {}, got {} instead'.format(self._numberOfMatrixEntries, len(U)))
        if type(self._matrix[0, 0]) != type(U[0]):
            self._matrix = np.zeros((self._size, self._size), dtype = type(U[0]))
        k = 0
        for i in range(self._size):
            for j in range(i,self._size):
                self._matrix[i, j] = U[k]
                k += 1
        for i in range(self._size-1):
            self._matrix[i+1, i] = 1

    def characteristicPolynomial(self):
        """
        Function to calculate the characteristic polynomial of unit upper Hessenberg Toeplitz Bohemian matrix
        Output:
        (numpy.array) Array of coefficients of the characteristic polynomial (in descending order of power)
        """
        # This routine is special for unit upper Hessenberg matrices.
        # The cost is O(n^2)
        # for computing characteristic polynomials for this class.
        # P(m) = mu*P(m - 1) - add(h[k,m-1]*P(k), k = 0 .. m-1)
        # See https://doi.org/10.1016/j.laa.2020.03.037
        # We here use numpy.polynomial.Polynomial arithmetic
        # to simulate symbolic computation.  What is returned
        # is an array of coefficients (which can later be cast
        # as a tuple or Polynomial, as desired)
        c = np.zeros(self._size + 1, dtype=type(self._matrix[0,0]))
        c[self._size] = 1
        plist = []
        plist.append( np.polynomial.Polynomial([1]) )
        plist.append( np.polynomial.Polynomial([-self._matrix[0,0],1]) )
        mu = np.polynomial.Polynomial([0,1])
        for m in range(2,self._size+1):
            s = 0
            for i in range(m):
                 s += self._matrix[i,m-1]*plist[i]
            p = mu*plist[m-1] - s
            plist.append( p )
        #
        for k in range(self._size):
            c[k] = plist[self._size].coef[k]
        return c

# # Executable part of the code
# U = [-1, 1, -1, -1, 0, 1, 1, 1, -1, 1, 1, 1, 0, 0, 1, 0]
# A = Bohemian(4, U)
# M = A.getMatrix()
# print('Matrix:\n', M)
# print('Number of Matrix Entries:', A.getNumberOfMatrixEntries())
# print('Characteristic Polynomial:', A.characteristicPolynomial())
# print('Determinant:', A.determinant())
# print('Eigenvalues:', A.eig())
#
# print(' ')
#
# U = [-1, 1, -1, -1, 0, 1, 1, 1, -1, 1]
# A = Symmetric(4, U)
# M = A.getMatrix()
# print('Matrix:\n', M)
# print('Number of Matrix Entries:', A.getNumberOfMatrixEntries())
# print('Characteristic Polynomial:', A.characteristicPolynomial())
# print('Determinant:', A.determinant())
# print('Eigenvalues:', A.eig())
