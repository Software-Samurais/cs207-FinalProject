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

def test_AD_repr():
    x = AD.AutoDiff(1.0)
    assert repr(x) == 'Function value: 1.0, Derivative value: 1.0'

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
    assert v._val == 1.0/3.0 and v._der == -0.7/3.0**2, "error with truediv"
    assert z._val == 1.0/(2*3.0) and z._der == -1.4/36, "error with truediv"


def test_rtruediv():
    x = AD.AutoDiff(3.0)
    y = 3.0 / x
    u = AD.AutoDiff(1.0, 2.0)
    v = x / u
    z = (2 * x) / u
    assert y._val == 1.0 and y._der == -1.0/3.0, "error with rtruediv"
    assert v._val == 3.0 and v._der == -5.0, "error with rtruediv"
    assert z._val == 6.0 and z._der == -10.0, "error with rtruediv"


def test_pow():
    x = AD.AutoDiff(2.0)
    y = x**3
    u = AD.AutoDiff(3.0, 4.0)
    z = u**2
    v = AD.AutoDiff(2.0, 0.5)
    w = v**4
    assert y._val == 8.0 and y._der == 12.0, "error with pow"
    assert z._val == 9.0 and z._der == 24.0, "error with pow"
    assert w._val == 16.0 and w._der == 16.0, "error with pow"

def test_AD_check_tol():
    x = AD.AutoDiff(np.pi/4)
    y = AD.check_tol(AD.tan(x))
    assert y._val == 1.0 and y._der == 2.0, "error with check_tol"

def test_exp():
    x = AD.AutoDiff(1.0, 2.0)
    y = AD.exp(x)
    u = AD.AutoDiff(2.0, 3.0)
    z = AD.exp(u) * AD.exp(x)
    assert y._val == np.exp(1.0) and y._der == 2.0*np.exp(1.0), "error with exp"
    assert z._val == np.exp(3.0) and np.abs(z._der - 5.0*np.exp(3.0)) < 10e-8, "error with exp"

def test_log():
    x = AD.AutoDiff(2.0, 3.0) 
    y = AD.log(x)
    assert y._val == np.log(x._val) and y._der == 1.5, "error with log"
    with pytest.raises(ValueError) as e:
        u = AD.AutoDiff(0)
        z = AD.log(u)
    assert str(e.value) == 'Cannot divide by Zero'
    
def test_sin():
    x = AD.AutoDiff(np.pi/2)
    y = AD.sin(x)
    u = AD.AutoDiff(np.pi/3, 2)
    z = AD.sin(u)
    assert y._val == 1.0 and y._der == 0.0, "error with sin"
    assert np.abs(z._val - np.sin(np.pi/3)) < 10e-8 and np.abs(z._der - 1.0)< 10e-8, "error with sin"

def test_cos():
    x = AD.AutoDiff(np.pi)
    y = AD.cos(x)
    u = AD.AutoDiff(np.pi/6, 2)
    z = AD.cos(u)
    assert y._val == -1.0 and y._der == 0.0, "error with cos"
    assert np.abs(z._val - np.cos(np.pi/6)) < 10e-8 and np.abs(z._der - (-1.0))< 10e-8, "error with cos"

def test_tan():
    x = AD.AutoDiff(np.pi/4)
    y = AD.tan(x)
    assert y._val == 1.0 and y._der == 2.0, "error with tan"
    with pytest.raises(ValueError) as e:
        u = AD.AutoDiff(np.pi/2)
        z = AD.tan(u)
    assert str(e.value) == 'Cannot divide by Zero'




