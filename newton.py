'''newton.py

Implementation of a Newton-Raphson root-finder.

'''

import numpy as np
import functions as F

class Newton(object):
    """Newton objects have a solve() method for finding roots of f(x)
    using Newton's method. x and f can both be vector-valued.

    """
    
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, Df=None, max_radius=None):
        """Parameters: f(function), tol(tolerance), maxiter, dx(step size), 
        Df(analytical jacobian, optional), max_radius(max radius from x0
        
        """
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx

        # set Df if specified, then analytical jacobian is calculated
        # otherwise approximate jacobian is calculated
        if Df != None:
            self._Df = Df
        else: self._Df = 0

        # set max_radius if specified, otherwise default max_radius=5
        if max_radius != None:
            self._max_radius = max_radius
        else:
            self._max_radius = 5



    def solve(self, x0):
        """Determine a solution of f(x) = 0, using Newton's method, starting
        from initial guess x0.

        """
        x = x0
        for i in range(self._maxiter):

            # evaluate f at x
            fx = self._f(x) 

            # if fx is close enough to 0, exit loop and return solution x
            if np.linalg.norm(fx) < self._tol:
                return x

            # iterate another step
            x = self.step(x, fx)

            # Bound the root: allow user to specify a radius max_radius around initial guess x0
            # returns 0 (with error message) if computed root is far away from x0
            if np.linalg.norm(x-x0) > self._max_radius: #pick a suitable  max_radius, default = 10
                print("computed root is above maximum radius threshold")
                return 0

        # Error if not converged after max interations
        if np.linalg.norm(fx) > self._tol:
            print( "Did not converge in maximum number of iterations. Change initial guess or increase number of iterations or tolerance")
            return 0

        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x. If the
        argument fx is provided, assumes fx = f(x).

        """
        Df_x = F.approximateJacobian(self._f, x, self._dx)

        # Determine if analytic form or approximate form of Jacobian is to be used
        if fx is None:
            fx = self._f(x)

        # Df_x^-1 f(x) is solved
        h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))

        # if x is a scalar, change h to scalar before 
        if np.isscalar(x):
            h = np.asscalar(h)

        return x - h
