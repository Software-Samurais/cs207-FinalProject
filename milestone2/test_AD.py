import doctest
import numpy as np
import pytest

import AD

def test_AD_val():
    
    x = AD.AutoDiff(1.0)
    
    assert x.val == 1.0
    
def test_AD_der():
    
    x = AD.AutoDiff(1.0)
    y = AD.AutoDiff(1.0, 0.1)
    
    assert x.der == 1.0
    
    assert y.der == 0.1

# This test needs to be fixed. Ideally, we should be comparing strings here. 
def test_AD_repr():
    
    x = AD.AutoDiff(1.0)
    
    assert print(x) == None
    
def test_AD_add():

    x = AD.AutoDiff(1.0)
    y = x+3
    z = 3+x
    
    u = AD.AutoDiff(3.0, 0.1)
    v = x + u
    
    assert y._val == 4.0, y._der == 1.0
    
    assert z._val == 4.0, z._der == 1.0
    
    assert v._val == 4.0, v._der == 1.1
    
def test_AD_sub():

    x = AD.AutoDiff(1.0)
    y = x-3
    z = 3-x
    
    u = AD.AutoDiff(3.0, 0.1)
    v = x-u
    
    assert y._val == -2.0, y._der == 1.0
    
    assert z._val == 2.0, z._der == 1.0
    
    assert v._val == -2.0, v._der == 0.9
    
def test_AD_mul():
    
    x = AD.AutoDiff(1.0)
    y = 3.0*x
    z = x*3.0
    
    u = AD.AutoDiff(2.0, 0.1)
    v = x*u
    
    assert y._val == 3.0, y._der == 3.0
    
    assert z._val == 3.0, z._der == 3.0
    
    assert v._val == 2.0, v._der == 2.1
    
def test_AD_div():

    x = AD.AutoDiff(3.0)
    y = x/3.0
    
    u = AD.AutoDiff(1.0)
    v = u/x
    
    assert y._val == 1.0, y._der == 1.0/3.0
    
    assert v._val == 1.0/3.0, v._der == 2.0/3.0**2
    
def test_AD_pow():
    
    x = AD.AutoDiff(2.0)
    y = x**2
    
    assert y._val == 4.0, y._der == 4.0
    
def test_AD_check_tol():
    
    x = AD.AutoDiff(np.pi/4)
    y = AD.check_tol(AD.tan(x))
    
    assert y._val == 1.0, y._der == 2.0
    
def test_AD_sin():
    
    x = AD.AutoDiff(np.pi/2)
    y = AD.sin(x)
    
    assert y._val == 1.0, y._der == 0.0
    
def test_AD_cos():
    
    x = AD.AutoDiff(np.pi)
    y = AD.cos(x)
    
    assert y._val == -1.0, y._der == 0.0
    
def test_AD_tan():
    
    x = AD.AutoDiff(np.pi/4)
    y = AD.tan(x)
    
    assert y._val == 1.0, y._der == 2.0
    
def test_AD_exp():
    
    x = AD.AutoDiff(1.0, 2.0)
    y = AD.exp(x)
    
    assert y._val == np.exp(1.0), y._der == 2.0*np.exp(1.0)
    
def test_AD_log():
    
    x = AD.AutoDiff(1.0) 
    y = AD.log(x)
    
    assert y._val == np.log(x._val), y._der == 0.0

