import numpy as np

def approximateJacobian(f, x, dx=1e-6):
    """Calculate a numerical approximation of the Jacobian Df(x).

    Parameters: f, x
    Returns: Df_x: numerical approximation to the Jacobian of f at x
    
    """
    # evaluate f at x (could be scalar or array)
    fx = f(x)

    # if f is scalar 1D, simply take the derivative of f
    if np.isscalar(x):
        return (f(x + dx) - fx) / dx

    # if f is not scalar, need to calculate Jacobian of all elements in f
    # initiate Jacobian array Df_x and step size array h
    N = x.size
    Df_x = np.matrix(np.zeros((N,N)))
    h = np.matrix(np.zeros_like(x,dtype=float))

    for i in range(x.size):
        h[i] = dx
        # Replace ith col of Df_x with difference quotient
        Df_x[:,i] = (f(x + h) - fx) / dx
        # Reset h[i] to 0
        h[i] = 0

    return Df_x


def AnalyticJacobian(f,x):
    Df_x=f(x)
    return Df_x


class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 5x + 6,
    and evaluate p(3):

    p = Polynomial([6, 5, 1])
    p(3)

    """
    def __init__(self, coeffs):
        """In coeffs, index = degree of that coefficient"""
        self._coeffs = coeffs

    #printable representation of the object in this class
    def __repr__(self):
        coeffstr = ",".join([str(x) for x in self._coeffs])
        return "Polynomial([{}])".format(coeffstr)

    def _f(self,x):
        ans = 0
        for c in reversed(self._coeffs):
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self._f(x)

