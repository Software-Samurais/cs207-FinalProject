import autodiff.AD as AD
import numpy as np
import math
import pytest


def test_val():
    x = AD.AutoDiff(1.0)
    assert x.val == 1.0, "error with val"
    
def test_der():
    x = AD.AutoDiff(1.0)
    y = AD.AutoDiff(1.0, 0.1)
    assert x.der == 1.0, "error with der"
    assert y.der == 0.1, "error with der"

def test_neg():
    x = AD.AutoDiff(1.0, 0.1)
    assert (-x).val == -1, "error with neg"
    assert (-x).der == -0.1, "error with neg"

def test_add():
    x = AD.AutoDiff(1.0)
    y = x + 3
    u = AD.AutoDiff(3.0, 0.1)
    v = x + u
    assert y._val == 4.0 and y._der == 1.0, "error with add"
    assert v._val == 4.0 and v._der == 1.1, "error with add"

def test_radd():
    x = AD.AutoDiff(1.0)
    y = 3 + x
    u = AD.AutoDiff(3.0, 0.1)
    v = u + x
    assert y._val == 4.0 and y._der == 1.0, "error with radd"
    assert v._val == 4.0 and v._der == 1.1, "error with radd"
"""
def test_sub():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x1-x2-3
    assert f.x == -4, "error with sub"
    assert (f.dx == np.array([1,-1,0])).all(), "error with sub"

def test_rsub():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 11-x1-x3
    assert f.x == 1, "error with rsub"
    assert (f.dx == np.array([-1,0,-1])).all(), "error with rsub"

def test_mul():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x1*x2*x3*2
    assert f.x == 240, "error with mul"
    assert (f.dx == np.array([60., 48., 40.])).all(), "error with mul"

def test_rmul():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 2*x1*x2*x3
    assert f.x == 240, "error with rmul"
    assert (f.dx == np.array([60., 48., 40.])).all(), "error with rmul"

def test_truediv():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x2/x1
    f2 = x2/2
    f3 = x2/(2*x1)
    f4 = x2/(x1*x1)
    assert f.x == 1.25, "error with truediv"
    assert (f.dx == np.array([-0.3125,  0.25  ,  0.    ])).all(), "error with truediv"
    assert f2.x == 2.5, "error with truediv"
    assert (f2.dx == np.array([0. , 0.5, 0. ])).all(), "error with truediv"
    assert f3.x == 0.625, "error with truediv"
    assert (f3.dx == np.array([-0.15625,  0.125  ,  0.])).all(), "error with truediv"
    assert f4.x == 0.3125, "error with truediv"
    assert (f4.dx == np.array([-0.15625,  0.0625 ,  0.])).all(), "error with truediv"

def test_rtruediv():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 1/x1
    f2 = 1/(2*x1)
    f3 = 7/(x1*x1)
    assert f.x == 0.25, "error with rtruediv"
    assert (f.dx == np.array([-0.0625, -0.    , -0.    ])).all(), "error with rtruediv"
    assert f2.x == 0.125, "error with rtruediv"
    assert (f2.dx == np.array([-0.03125, -0.     , -0.   ])).all(), "error with rtruediv"
    assert f3.x == 0.4375, "error with rtruediv"
    assert (f3.dx == np.array([-0.21875, -0.     , -0.     ])).all(), "error with rtruediv"

def test_pow():
    sol=vt.Solver(2)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    f = (x1+x2) ** 2
    assert f.x == 81, "error with pow"
    assert (f.dx == np.array([18., 18.])).all(), "error with pow"

def test_exp():
    sol=vt.Solver(2)
    x1=sol.create_variable(0)
    x2=sol.create_variable(5)
    f = np.exp(x1) + x2
    assert f.x == 6.0, "error with exp"
    assert (f.dx == np.array([1., 1.])).all(), "error with exp"

def test_log():
    sol=vt.Solver(2)
    x1=sol.create_variable(10)
    x2=sol.create_variable(5)
    f = np.log(x1) + np.log(x2)
    assert f.x == 3.9120230054281464, "error with log"
    assert (f.dx == np.array([0.1, 0.2])).all(), "error with log"

def test_sin():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.sin(x1) + np.sin(x2)
    assert f.x == 1.5, "error with sin"
    assert ((f.dx - np.array([6.12323400e-17, 8.66025404e-01])) < 10**(-8)).sum() == 2, "error with sin"

def test_cos():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.cos(x1) + np.cos(x2)
    assert f.x == 0.8660254037844388, "error with cos"
    assert ((f.dx - np.array([-1.,-0.5])) < 10**(-8)).sum() == 2, "error with cos"

def test_tan():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.tan(x1) + np.tan(x2)
    assert f.x == 1.633123935319537e+16, "error with tan"
    assert ((f.dx - np.array([2.66709379e+32, 1.33333333e+00])) < 10**(-8)).sum() == 2, "error with tan"


"""
