"""
This module provides a collection of common utility functions for numerical calculations,
including matrix operations, interpolation, and special polynomials.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import lapack


class Common:
    """
    A collection of static methods for common numerical tasks.
    """

    @staticmethod
    def MatInv(matin : np.ndarray) -> np.ndarray:
        """
        Computes the inverse of a complex square matrix.

        Args:
            matin (np.ndarray): A complex-valued square matrix of shape (n, n).

        Returns:
            np.ndarray: The inverse of the input matrix.
        """

        if (matin.ndim != 2 or matin.shape[0] != matin.shape[1]):
            raise ValueError("Input must be a square 2D matrix.")
        
        if (matin.dtype != np.complex128):
            matin = matin.astype(np.complex128)
        
        tempmat = np.copy(matin)

        lu, piv, info = lapack.zgetrf(tempmat, overwrite_a=True)

        if (info < 0):
            raise np.linalg.LinAlgError(
            f"LAPACK error (zgetrf): Illegal value in argument {-info}"
            )
        elif (info > 0):
            raise np.linalg.LinAlgError(
            "LAPACK error (zgetrf): Matrix is singular."
            )
        
        invmat, info = lapack.zgetri(lu, piv, overwrite_lu=True)

        if (info < 0):
            raise np.linalg.LinAlgError(
            f"LAPACK error (zgetri): Illegal value in argument {-info}"
            )
        
        return invmat
    
    @staticmethod
    def Indexing(ntot, ndivision, divisionarray, flag, n1, n2):
        """Map between a flat 1D index and a multi-dimensional index.

        Args:
            ntot (int): Total number of elements in the multi-dimensional array.
            ndivision (int): The number of dimensions.
            divisionarray (list of int): A list containing the size of each dimension.
            flag (int): Conversion direction flag. 
                        1: Encode from multi-dimensional index (n2) to flat index (n1).
                        0: Decode from flat index (n1) to multi-dimensional index (n2).
            n1 (int): The flat (1D) index. Input for decoding, output for encoding.
            n2 (list of int): The multi-dimensional index. Input for encoding, output for decoding.

        Returns:
            tuple: A tuple (n1, n2) containing the updated flat and multi-dimensional indices.
        """
        tmpsize = 1
        for size in divisionarray:
            tmpsize *= size

        if tmpsize != ntot:
            print('array_division wrong')
            return

        if flag == 1:
            n1 = n2[0]
            for ii in range(1, ndivision):
                tempcnt = 1
                for jj in range(ii):
                    tempcnt *= divisionarray[jj]
                n1 += (n2[ii] ) * tempcnt
        else:
            n2_array = [0] * ndivision
            tempcnt = n1
            for ii in range(ndivision - 1):
                n2_array[ii] = tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
                tempcnt = (tempcnt - n2_array[ii])//divisionarray[ii]
            n2_array[ndivision - 1] = tempcnt

            # Copy the values from the temporary array to the n2 output array
            for i in range(ndivision):
                n2[i] = n2_array[i]

        return n1, n2
    
    @staticmethod
    def HermitianEigenCmplx(datamat : np.ndarray):
        """
        Computes the eigenvalues and eigenvectors of a complex Hermitian matrix.

        Args:
            datamat (np.ndarray): A complex Hermitian matrix.

        Returns:
            tuple: A tuple containing:
                - w (np.ndarray): The eigenvalues of the matrix.
                - v (np.ndarray): The eigenvectors of the matrix.
        """
        w, v = eigh(datamat)
        return w, v
    
    @staticmethod
    def FderivCmplx(m, x, f):
        """
        Computes the m-th derivative or anti-derivative of a complex-valued function.

        Args:
            m (int): The order of the derivative. Negative values correspond to anti-derivatives.
            x (np.ndarray): The array of x-coordinates (abscissa).
            f (np.ndarray): The array of complex function values at x.

        Returns:
            np.ndarray: The computed (anti-)derivative of the function.
        """
        n = len(x)
        g = np.zeros_like(f, dtype=complex)
        
        if m == -3:
            # Low accuracy trapezoidal integration
            g[0] = 0.0
            for i in range(n-1):
                g[i+1] = g[i] + 0.5 * (x[i+1] - x[i]) * (f[i+1] + f[i])
            return g
        
        elif m == -2:
            # Medium accuracy Simpson integration
            g[0] = 0.0
            for i in range(n-2):
                x0, x1, x2 = x[i], x[i+1], x[i+2]
                g[i+1] = g[i] + (x0-x1) * (
                    f[i+2] * (x0-x1)**2 +
                    f[i+1] * (x2-x0) * (x0+2*x1-3*x2) +
                    f[i] * (x2-x1) * (2*x0+x1-3*x2)
                ) / (6 * (x0-x2) * (x1-x2))
            
            # Last point
            x0, x1, x2 = x[n-1], x[n-2], x[n-3]
            g[n-1] = g[n-2] + (x1-x0) * (
                f[n-3] * (x1-x0)**2 +
                f[n-1] * (x1-x2) * (3*x2-x1-2*x0) +
                f[n-2] * (x0-x2) * (3*x2-2*x1-x0)
            ) / (6 * (x2-x1) * (x2-x0))
            return g
        
        elif m == 0:
            return f.copy()
        
        elif m >= 4:
            return np.zeros_like(f)
        
        else:
            # High accuracy spline interpolation
            cf = Common.SplineCmplx(x, f)
            
            if m <= -1:
                # Integration
                g[0] = 0.0
                for i in range(n-1):
                    dx = x[i+1] - x[i]
                    g[i+1] = g[i] + (((0.25*cf[2,i]*dx + 
                                      0.3333333333333333*cf[1,i])*dx + 
                                     0.5*cf[0,i])*dx + f[i])*dx
            elif m == 1:
                g = cf[0, :]
            elif m == 2:
                g = 2.0 * cf[1, :]
            elif m == 3:
                g = 6.0 * cf[2, :]
            
            return g

    @staticmethod
    def SplineCmplx(x, f):
        """
        Calculates the coefficients for a cubic spline interpolation of complex data.

        Args:
            x (np.ndarray): The array of x-coordinates (abscissa).
            f (np.ndarray): The array of complex function values at x.

        Returns:
            np.ndarray: An array of shape (3, n) containing the cubic spline coefficients.
        """
        
        n = len(x)
        cf = np.zeros((3, n), dtype=np.complex128)
        
        if n == 1:
            cf[:, 0] = 0.0
            return cf
        
        if n == 2:
            cf[0, 0] = (f[1] - f[0]) / (x[1] - x[0])
            cf[1:3, 0] = 0.0
            cf[0, 1] = cf[0, 0]
            cf[1:3, 1] = 0.0
            return cf
        
        if n == 3:
            x0 = x[0]
            x1 = x[1] - x0
            x2 = x[2] - x0
            y0 = f[0]
            y1 = f[1] - y0
            y2 = f[2] - y0
            t0 = 1.0 / (x1 * x2 * (x2 - x1))
            t1 = x1 * y2
            t2 = x2 * y1
            c1 = t0 * (x2 * t2 - x1 * t1)
            c2 = t0 * (t1 - t2)
            cf[0, 0] = c1
            cf[1, 0] = c2
            cf[2, 0] = 0.0
            t3 = 2.0 * c2
            cf[0, 1] = c1 + t3 * x1
            cf[1, 1] = c2
            cf[2, 1] = 0.0
            cf[0, 2] = c1 + t3 * x2
            cf[1, 2] = c2
            cf[2, 2] = 0.0
            return cf
        
        y0 = f[0]
        y1 = f[1] - y0
        y2 = f[2] - y0
        y3 = f[3] - y0
        x0 = x[0]
        x1 = x[1] - x0
        x2 = x[2] - x0
        x3 = x[3] - x0
        t0 = 1.0 / (x1 * x2 * x3 * (x1 - x2) * (x1 - x3) * (x2 - x3))
        t1 = x1 * x2 * y3
        t2 = x2 * x3 * y1
        t3 = x3 * x1 * y2
        t4 = x1**2
        t5 = x2**2
        t6 = x3**2
        y1 = t3 * t6 - t1 * t5
        y3 = t2 * t5 - t3 * t4
        y2 = t1 * t4 - t2 * t6
        c1 = t0 * (x1 * y1 + x2 * y2 + x3 * y3)
        c2 = -t0 * (y1 + y2 + y3)
        c3 = t0 * (t1 * (x1 - x2) + t2 * (x2 - x3) + t3 * (x3 - x1))
        cf[0, 0] = c1
        cf[1, 0] = c2
        cf[2, 0] = c3
        cf[0, 1] = c1 + 2.0 * c2 * x1 + 3.0 * c3 * t4
        cf[1, 1] = c2 + 3.0 * c3 * x1
        cf[2, 1] = c3
        
        if n == 4:
            cf[0, 2] = c1 + 2.0 * c2 * x2 + 3.0 * c3 * t5
            cf[1, 2] = c2 + 3.0 * c3 * x2
            cf[2, 2] = c3
            cf[0, 3] = c1 + 2.0 * c2 * x3 + 3.0 * c3 * t6
            cf[1, 3] = c2 + 3.0 * c3 * x3
            cf[2, 3] = c3
            return cf
        
        for i in range(2, n-2):
            y0 = f[i]
            y1 = f[i-1] - y0
            y2 = f[i+1] - y0
            y3 = f[i+2] - y0
            x0 = x[i]
            x1 = x[i-1] - x0
            x2 = x[i+1] - x0
            x3 = x[i+2] - x0
            t1 = x1 * x2 * y3
            t2 = x2 * x3 * y1
            t3 = x3 * x1 * y2
            t0 = 1.0 / (x1 * x2 * x3 * (x1 - x2) * (x1 - x3) * (x2 - x3))
            c3 = t0 * (t1 * (x1 - x2) + t2 * (x2 - x3) + t3 * (x3 - x1))
            t4 = x1**2
            t5 = x2**2
            t6 = x3**2
            y1 = t3 * t6 - t1 * t5
            y2 = t1 * t4 - t2 * t6
            y3 = t2 * t5 - t3 * t4
            cf[0, i] = t0 * (x1 * y1 + x2 * y2 + x3 * y3)
            cf[1, i] = -t0 * (y1 + y2 + y3)
            cf[2, i] = c3
        
        c1 = cf[0, n-3]
        c2 = cf[1, n-3]
        c3 = cf[2, n-3]
        cf[0, n-2] = c1 + 2.0 * c2 * x2 + 3.0 * c3 * t5
        cf[1, n-2] = c2 + 3.0 * c3 * x2
        cf[2, n-2] = c3
        cf[0, n-1] = c1 + 2.0 * c2 * x3 + 3.0 * c3 * t6
        cf[1, n-1] = c2 + 3.0 * c3 * x3
        cf[2, n-1] = c3
        return cf

    @staticmethod
    def BernoulliPolynomial(x, n):
        """
        Computes the n-th Bernoulli polynomial at a given point x.

        Args:
            x (float): The point at which to evaluate the polynomial.
            n (int): The order of the Bernoulli polynomial (0 to 6).

        Returns:
            float: The value of the Bernoulli polynomial B_n(x).
        """

        val = 0.0
        xmat = np.array([x**6, x**5, x**4, x**3, x**2, x, 1.0])

        if n == 0:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        elif n == 1:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0/2.0]))
        elif n == 2:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0/6.0]))
        elif n == 3:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 1.0, -3.0/2.0, 1.0/2.0, 0.0]))
        elif n == 4:
            val = np.sum(xmat * np.array([0.0, 0.0, 1.0, -2.0, 1.0, 0.0, -1.0/30.0]))
        elif n == 5:
            val = np.sum(xmat * np.array([0.0, 1.0, -5.0/2.0, 5.0/3.0, 0.0, -1.0/6.0, 0.0]))
        elif n == 6:
            val = np.sum(xmat * np.array([1.0, -3.0, 5.0/2.0, 0.0, -1.0/2.0, 0.0, 1.0/42.0]))
        
        return val
    
    @staticmethod
    def EulerPolynomial(x, n):
        """
        Computes the n-th Euler polynomial at a given point x.

        Args:
            x (float): The point at which to evaluate the polynomial.
            n (int): The order of the Euler polynomial (0 to 6).

        Returns:
            float: The value of the Euler polynomial E_n(x).
        """

        
        val = 0.0
        
        xmat = np.array([x**6, x**5, x**4, x**3, x**2, x, 1.0])

        if n == 0:
         val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        elif n == 1:
            val= np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0/2.0]))
        elif n == 2:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]))
        elif n == 3:
            val = np.sum(xmat * np.array([0.0, 0.0, 0.0, 1.0, -3.0/2.0, 0.0, 1.0/4.0]))
        elif n == 4:
            val = np.sum(xmat * np.array([0.0, 0.0, 1.0, -2.0, 0.0, 1.0, 0.0]))
        elif n == 5:
            val = np.sum(xmat * np.array([0.0, 1.0, -5.0/2.0, 0.0, 5.0/2.0, 0.0, -1.0/2.0]))
        elif n == 6:
            val = np.sum(xmat * np.array([1.0, -3.0, 0.0, 5.0, 0.0, -3.0, 0.0]))
        
        return val

    @staticmethod
    def FactorialInt(j):
        """
        Computes the factorial of a non-negative integer.

        Args:
            j (int): A non-negative integer.

        Returns:
            int: The factorial of j.

        Raises:
            ValueError: If j is a negative integer.
        """

        if j < 0:
            raise ValueError("factorial is defined only for non-negative numbers")
        x = 1
        num = x
        if (j == 1):
            return num
        else:
            for i in range(2, j+1):
                x = x*i
            num = x
            return num
        
    @staticmethod
    def Ttind(itheta, ntau):
        """
        Performs an index transformation for tau-theta grids.

        Args:
            itheta (int): The index in the theta grid.
            ntau (int): The number of points in the tau grid.

        Returns:
            int: The transformed index in the tau grid.
        """
        if itheta >= 0:
            return ntau - 1 - itheta
        else:
            return -ntau - 1 - itheta
        
    @staticmethod
    def Gcoeff(m):
        """
        Calculates the coefficient for the Legendre-Chebyshev transformation.

        Args:
            m (int): A non-negative integer.

        Returns:
            float: The calculated coefficient.

        Raises:
            ValueError: If m is a negative integer.
        """
        if m < 0:
            raise ValueError("gcoeff defined only for non-negative numbers!")
        
        if m == 0:
            return 1.0
        else:
            result = 1.0
            # Calculate (2m-1)!! / (2^m * m!)
            for ii in range(1, m + 1):
                result *= (2 * ii - 1)
            for ii in range(1, m + 1):
                result /= (ii * 2)
            return result
        
    @staticmethod
    def MinDistance(S, d):
        """
        Finds the minimum distance to a point in a lattice, considering periodic boundary conditions.

        Args:
            S (np.ndarray): A 3x3 matrix where rows are the lattice vectors.
            d (np.ndarray): A 3D position vector.

        Returns:
            float: The minimum distance R.
        """
        R = 1.0e6
        R1 = np.sqrt(np.sum(d**2))
        
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    rr = np.zeros((3, 3))
                    rr[0, :] = ix * S[0, :]
                    rr[1, :] = iy * S[1, :]
                    rr[2, :] = iz * S[2, :]
                    
                    dtemp = d - rr[0, :] - rr[1, :] - rr[2, :]
                    R2 = np.sqrt(np.sum(dtemp**2))
                    
                    Rtemp = min(R1, R2)
                    
                    if Rtemp <= R:
                        R = Rtemp
        
        return R

    @staticmethod
    def KIdx2KVec(grid : list) -> dict:

        nk = grid[0]*grid[1]*grid[2]
        kidx = {}
        for ik in range(nk):
            n1, n2 = Common.Indexing(nk, 3, grid, 0, ik, [0, 0, 0])
            # print(n1, n2)
            kidx[ik] = n2

        return kidx

    @staticmethod
    def KVec2KIdx(klist : list, kidx : dict) -> int:
        
        for key, val in kidx.items():
            if val == klist:
                return key
    
    @staticmethod
    def FindPositions(array, value):
        """Find all positions of a value in a 2D array.

        Args:
            array (iterable of iterable): 2D array to search.
            value: Value to search for.

        Returns:
            list of [int, int]: List of [row_index, col_index] where value matches.
        """
        positions = []
        for row_index, row in enumerate(array):
            for col_index, col_value in enumerate(row):
                if col_value == value:
                    positions.append([row_index, col_index])
        return positions