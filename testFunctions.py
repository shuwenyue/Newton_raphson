#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt
import functions as F

class TestFunctions(unittest.TestCase):

    # calculation of jacobian for 1D function using ApproxJacobian
    def test_ApproxJacobian1D(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.approximateJacobian(f, x0, dx) # calculate derivative
        self.assertTrue(np.isscalar(Df_x)) # check Jacobian is a scalar
        self.assertAlmostEqual(Df_x, slope)

    # calculate jacobian for 2D function using ApproxJacobian
    def test_ApproxJacobian2D(self):
        # u1 = x1 + 2 x2
        # u2 = 3 x1 + 4 x2
        A = np.matrix("1.0 2.0; 3.0 4.0")
        def f(x):
            return A * x
        x0 = np.matrix("5.0; 6.0")
        dx = 1.e-6
        Df_x = F.approximateJacobian(f, x0, dx) # calculate Jacobian array
        self.assertEqual(Df_x.shape, (2,2)) # check size of Jacobian array shape
        npt.assert_array_almost_equal(Df_x, A)

    # test Polynomial class 
    def test_Polynomial(self):
        # p(x) = x^2 + 5x + 4
        p = F.Polynomial([4, 5, 1])
        for x in np.linspace(-2,2,11):
            self.assertAlmostEqual(p(x), 4 + 5*x + x**2)

    # pass in Polynomial object to ApproxJacobian
    def test_ApproxJacobianPolynomial(self):
        # p(x) = x^2 + 5x + 4
        p = F.Polynomial([4, 5, 1])
        x0=0
        dx=1e-8
        Df_x = F.approximateJacobian(p, x0,dx)
        self.assertAlmostEqual(Df_x,5.0)

    # test analytic jacobial function
    def testAnalytic(self):
        # Df= 2x + 5
        Df = F.Polynomial([5, 2])
        x0 = -1
        Df_x = F.AnalyticJacobian(Df, x0)
        np.testing.assert_array_almost_equal(Df_x,3)

if __name__ == '__main__':
    unittest.main()



