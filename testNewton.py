#!/usr/bin/env python3

import unittest
import numpy as np
import functions as F
import newton
import math

class TestNewton(unittest.TestCase):

    # find root of simple linear function
    def testLinear(self):
        f = lambda x : 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x0 = -2.0
        x = solver.solve(x0)
        self.assertEqual(x, -2.0)

    # check root of nonlinear function
    def testNonLinear(self):
        f = lambda x : x*x-1
        solver = newton.Newton(f, tol=1.e-15, maxiter=30)
        x0 = 0.5
        x = solver.solve(x0)    
        self.assertEqual(x, 1.0)

    # check 'bound the root' if x step is > max_radius
    def testBoundtheRoot(self):
        f = lambda x: 3.0*x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2,max_radius=2)
        x0 = 100
        x = solver.solve(x0)
        self.assertEqual(x, 0) # 0 return value for 'bound the root' error

    # check max iterations
    def testMaxIterations(self):
        f = F.Polynomial([12,-7,1])
        solver = newton.Newton(f, tol=1.e-15, maxiter=10,max_radius=100)
        x0 = 90
        x = solver.solve(x0)
        self.assertEqual(x, 0) # 0 return value for max iteration error


        

    

if __name__ == "__main__":
    unittest.main()

    
