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

def test_sub():
    x = AD.AutoDiff(1.0)
    y = x - 3
    u = AD.AutoDiff(3.0, 0.1)
    v = x - u
    assert y._val == -2.0 and y._der == 1.0, "error with sub"
    assert v._val == -2.0 and v._der == 0.9, "error with sub"

def test_rsub():
    x = AD.AutoDiff(1.0)
    y = 3 - x
    u = AD.AutoDiff(3.0, 0.1)
    v = u - x
    assert y._val == 2.0 and y._der == -1.0, "error with rsub"
    assert v._val == 2.0 and v._der == -0.9, "error with rsub"

def test_mul():
    x = AD.AutoDiff(1.0)
    y = 3.0 * x
    u = AD.AutoDiff(2.0, 0.1)
    v = x * u
    z = u * (3*x)
    assert y._val == 3.0 and y._der == 3.0, "error with mul"
    assert v._val == 2.0 and v._der == 2.1, "error with mul"
    assert z._val == 6.0 and z._der == 6.3, "error with mul"

def test_rmul():
    x = AD.AutoDiff(1.0)
    y = x * 3.0
    u = AD.AutoDiff(2.0, 0.1)
    v = u * x
    z = (u*3) * x
    assert y._val == 3.0 and y._der == 3.0, "error with rmul"
    assert v._val == 2.0 and v._der == 2.1, "error with rmul"
    assert z._val == 6.0 and z._der == 6.3, "error with rmul"
   
def test_truediv():
    x = AD.AutoDiff(3.0)
    y = x / 3.0
    u = AD.AutoDiff(1.0, 0.1)
    v = u / x
    z = u / (2 * x)
    assert y._val == 1.0 and y._der == 1.0/3.0, "error with truediv"
    assert v._val == 1.0/3.0 and v._der == -0.7/9, "error with truediv"
    assert z._val == 1.0/(2*3.0) and z._der == -1.4/36, "error with truediv"


def test_rtruediv():
    x = AD.AutoDiff(3.0)
    y = 3.0 / x
    u = AD.AutoDiff(1.0, 2.0)
    v = x / u
    z = (2 * x) / u
    assert y._val == 1.0 and y._der == -1.0/3.0, "error with truediv"
    assert v._val == 3.0 and v._der == -5.0, "error with truediv"
    assert z._val == 6.0 and z._der == -10.0, "error with truediv"
"""
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
