import doctest
import numpy as np
import pytest

import AD

def test_AD_add():

    x = AD.AutoDiff(1.0)
    y = x+3
    z = 3+x
    
    u = AD.AutoDiff(3.0, 0.1)
    v = x + u
    
    assert y.val == 4.0, y.der == 1.0
    
    assert z.val == 4.0, z.der == 1.0
    
    assert v.val == 4.0, v.der == 1.1
    
def test_AD_sub():

    x = AD.AutoDiff(1.0)
    y = x-3
    z = 3-x
    
    u = AD.AutoDiff(3.0, 0.1)
    v = x-u
    
    assert y.val == -2.0, y.der == 1.0
    
    assert z.val == 2.0, z.der == 1.0
    
    assert v.val == -2.0, v.der == 0.9
    
def test_AD_mul():
    
    x = AD.AutoDiff(1.0)
    y = 3.0*x
    z = x*3.0
    
    u = AD.AutoDiff(2.0, 0.1)
    v = x*u
    
    assert y.val == 3.0, y.der == 3.0
    
    assert z.val == 3.0, z.der == 3.0
    
    assert v.val == 2.0, v.der == 2.1
    
def test_AD_div():

    x = AD.AutoDiff(3.0)
    y = x/3.0
    
    u = AD.AutoDiff(1.0)
    v = u/x
    
    assert y.val == 1.0, y.der == 1.0/3.0
    
    assert v.val == 1.0/3.0, v.der == 2.0/3.0**2
    
def test_AD_pow():
    
    x = AD.AutoDiff(2.0)
    y = x**2
    
    assert y.val == 4.0, y.der == 4.0
    
def test_AD_check_tol():
    
    x = AD.AutoDiff(np.pi/4)
    y = AD.check_tol(AD.tan(x))
    
    assert y.val == 1.0, y.der == 2.0
    
def test_AD_sin():
    
    x = AD.AutoDiff(np.pi/2)
    y = AD.sin(x)
    
    assert y.val == 1.0, y.der == 0.0
    
def test_AD_cos():
    
    x = AD.AutoDiff(np.pi)
    y = AD.cos(x)
    
    assert y.val == -1.0, y.der == 0.0
    
def test_AD_tan():
    
    x = AD.AutoDiff(np.pi/4)
    y = AD.tan(x)
    
    assert y.val == 1.0, y.der == 2.0
    
def test_AD_exp():
    
    x = AD.AutoDiff(1.0, 2.0)
    y = AD.exp(x)
    
    assert y.val == np.exp(1.0), y.der == 2.0*np.exp(1.0)
