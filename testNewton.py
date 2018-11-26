#!/usr/bin/env python3

import unittest
import numpy as np
import functions as F
import newton
import math

class TestNewton(unittest.TestCase):

    # check single Newton step
    def test_NewtonStep(self):
        f = F.Polynomial([9,6,1])
        solver = newton.Newton(f, tol=1.e-10, maxiter=1)
        x0 = 0
        x = solver.solve(x0)
        self.assertAlmostEqual(x, -1.5,places=6) 
        trueroot = -3
        self.assertTrue(abs(trueroot-x)<abs(trueroot-x0))

    # find root of simple linear function
    def test_Linear(self):
        f = lambda x : 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x0 = -2.0
        x = solver.solve(x0)
        self.assertEqual(x, -2.0)

    # find root of nonlinear function
    def test_NonLinear(self):
        f = lambda x : x*x-1
        solver = newton.Newton(f, tol=1.e-15, maxiter=30)
        x0 = 0.5
        x = solver.solve(x0)    
        self.assertEqual(x, 1.0)

    # check 'bound the root' error if x step is > max_radius
    def test_BoundtheRoot(self):
        f = lambda x: 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2,max_radius=2)
        x0 = 100
        self.assertRaises(Exception,solver.solve,x0)

    # check max iterations error
    def test_MaxIterations(self):
        f = F.Polynomial([1,0,1]) # function with no root, never converges
        solver = newton.Newton(f, tol=1.e-15, maxiter=100,max_radius=100)
        x0 = 2
        self.assertRaises(Exception,solver.solve,x0)

    # check solver with 1D analytic jacobian
    def test_analytic_1d(self):
        f= F.Polynomial([12,-7,1])
        Df= lambda x: 2*x-7
        solver = newton.Newton(f, tol=1.e-15, Df=Df)
        x = solver.solve(3.6)
        self.assertAlmostEqual(x, 4.0)

    # check solver with 2D analytic jacobian
    def test_analytic_2d(self):
        f= lambda x: np.matrix([[2*x[0,0]-2*x[1,0]-2],[ x[0,0]+2*x[1,0]-7]])
        Df= lambda x: np.matrix([[2, -2],[1, 2]])
        solver = newton.Newton(f, tol=1.e-10, Df=Df)
        x0=np.matrix([[1.],[1.]])
        x = solver.solve(x0)
        np.testing.assert_array_almost_equal(x, np.matrix([[3], [2]]))

    # check solver with 1D with NO analytic jacobian
    def test_approximate_1d(self):
        f= F.Polynomial([12,-7,1])
        solver = newton.Newton(f, tol=1.e-15)
        x = solver.solve(3.6)
        self.assertAlmostEqual(x, 4.0)

    # check solver with 2D NO analytic jacobian
    def test_approximate_2d(self):
        f= lambda x: np.matrix([[2*x[0,0]-2*x[1,0]-2],[ x[0,0]+2*x[1,0]-7]])
        solver = newton.Newton(f, tol=1.e-10 )
        x0=np.matrix([[1.],[1.]])
        x = solver.solve(x0)
        np.testing.assert_array_almost_equal(x, np.matrix([[3], [2]]))

    # find root of quadratic function with a single root
    def test_Quadratic(self):
        f= F.Polynomial([9,6,1])
        Df=F.Polynomial([6,2])
        solver = newton.Newton(f, tol=1.e-15, Df=Df, maxiter=50)
        x = solver.solve(-3)
        self.assertAlmostEqual(x, -3.0)

    # raise error if guess is at slope 0 of function
    def test_SlopeZero(self):
        f= F.Polynomial([12,-7,1])
        Df=F.Polynomial([-7,2])
        solver = newton.Newton(f, tol=1.e-10, Df=Df, maxiter=50)
        x0=3.5
        self.assertRaises(Exception,solver.solve,x0)

if __name__ == "__main__":
    unittest.main()

    
