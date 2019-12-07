import autodiff.AD as AD
import numpy as np
import pytest

def test_init():
    x = AD.Var([1, 2, 3])
    y = AD.Var(1)
    z = AD.Var([1,2], [2,4])
    assert np.array_equal(x.val, [1, 2, 3]) and np.array_equal(x.der, np.ones(3)), "error with init"
    assert y.val == 1.0 and y.der == 1.0, "error with init"
    assert np.array_equal(z.val, [1, 2]) and np.array_equal(z.der, [2, 4]), "error with init"
    with pytest.raises(KeyError) as e:
        x = AD.Var(3, [1, 2])
    assert str(e.value) == "'The format of value and derivative is not align'"

    
def test_val():
    x = AD.Var([1], [1, 0, 0])
    y = AD.Var([2], [0, 1, 0])
    z = AD.Var([3], [0, 0, 1])
    u = AD.Var([x, y])
    f = x + y 
    assert x.val == 1.0, "error with val"
    assert np.array_equal(u.val, [1, 2]), "error with val"
    assert f.val == 3.0, "error with val"
 
def test_der():
    x = AD.Var([1], [1, 0, 0])
    y = AD.Var([2], [0, 1, 0])
    z = AD.Var([3], [0, 0, 1])
    f = x + y + z
    g = AD.Var([2 * x + x * y])
    x1 = AD.Var(1.0)
    y1 = AD.Var(1.0, 0.1)
    assert x1.der == 1.0, "error with der"
    assert y1.der == 0.1, "error with der"
    assert np.array_equal(f.der, [1, 1, 1]), "error with der"
    assert np.array_equal(g.der, [[4, 1, 0]]), "error with der"
    
def test_val_set():
    x = AD.Var(1.0)
    x.val = 2.0
    assert x.val == 2.0, "error with val setter"
    
def test_der_set():
    x = AD.Var(1.0)
    x.der = 2.0
    assert x.der == 2.0, "error with der setter"

def test__repr__():
    
    # Simple scalar variable
    x = AD.Var(1.0)
    assert repr(x) == 'Function value:\n1.0\nDerivative value:\n1.0'
    
    # Scalar variable with array inputs
    x = np.linspace(0, 1, 10)
    y = AD.Var(x)
    assert repr(y) == f"Function values:\n{x}\nDerivative values:\n{np.ones(x.size)}"
    
    # Vector variables
    x = AD.Var([1, 2, 3], [1, 0, 0])
    y = AD.Var([4, 5, 6], [0, 1, 0])
    assert repr(x+y) == f"Function values:\n{(x+y).val}\nDerivative values:\n{(x+y).der}"

    x = AD.Var([1], [1, 0, 0])
    y = AD.Var([2], [0, 1, 0])
    z = AD.Var([3], [0, 0, 1])
    f = x + y + z
    assert repr(f) == f"Function values:\n{f.val}\nGradient:\n{f.der}"
    g = AD.Var([x, y ** 2, z ** 4])
    assert repr(g) == f"Function values:\n{g.val}\nJacobian:\n{g.der}"
    
def test_neg():
    x = AD.Var(1.0, 0.1)
    assert (-x).val == -1, "error with neg"
    assert (-x).der == -0.1, "error with neg"

def test_add():
    x = AD.Var(1.0)
    y = x + 3
    u = AD.Var(3.0, 0.1)
    v = x + u
    assert y._val == 4.0 and y._der == 1.0, "error with add"
    assert v._val == 4.0 and v._der == 1.1, "error with add"
    a = AD.Var([3], [1, 0])
    b = AD.Var([4], [0, 1])
    z = a + b
    assert np.array_equal(z._val, [7]) and np.array_equal(z._der, [1.0, 1.0]), "error with add"

def test_radd():
    x = AD.Var(1.0)
    y = 3 + x
    u = AD.Var(3.0, 0.1)
    v = u + x
    assert y._val == 4.0 and y._der == 1.0, "error with radd"
    assert v._val == 4.0 and v._der == 1.1, "error with radd"
 
def test_sub():
    x = AD.Var(1.0)
    y = x - 3
    u = AD.Var(3.0, 0.1)
    v = x - u
    assert y._val == -2.0 and y._der == 1.0, "error with sub"
    assert v._val == -2.0 and v._der == 0.9, "error with sub"
    a = AD.Var([3], [1, 0])
    b = AD.Var([4], [0, 1])
    z = a - b
    assert np.array_equal(z._val, [-1]) and np.array_equal(z._der, [1.0, -1.0]), "error with add"

def test_rsub():
    x = AD.Var(1.0)
    y = 3 - x
    u = AD.Var(3.0, 0.1)
    v = u - x
    assert y._val == 2.0 and y._der == -1.0, "error with rsub"
    assert v._val == 2.0 and v._der == -0.9, "error with rsub"

def test_mul():
    x = AD.Var(1.0)
    y = 3.0 * x
    u = AD.Var(2.0, 0.1)
    v = x * u
    z = u * (3*x)
    assert y._val == 3.0 and y._der == 3.0, "error with mul"
    assert v._val == 2.0 and v._der == 2.1, "error with mul"
    assert z._val == 6.0 and z._der == 6.3, "error with mul"
    x = AD.Var([3.0], [1, 0])
    y = AD.Var([2], [0, 1])
    z = x * y
    assert np.array_equal(z._val, [6.0]) and np.array_equal(z._der, [2.0, 3.0]), "error with mul"


def test_rmul():
    x = AD.Var(1.0)
    y = x * 3.0
    u = AD.Var(2.0, 0.1)
    v = u * x
    z = (u*3) * x
    assert y._val == 3.0 and y._der == 3.0, "error with rmul"
    assert v._val == 2.0 and v._der == 2.1, "error with rmul"
    assert z._val == 6.0 and z._der == 6.3, "error with rmul"
    x = AD.Var([3.0], [1, 0, 0])
    y = AD.Var([1.0], [0, 1, 0])
    w = AD.Var([2.0], [0, 0, 1])
    z = x + y ** 2 + x * w
    assert np.array_equal(z._val, [10.0]) and np.array_equal(z._der, [3.0, 2.0, 3.0]), "error with rmul"

   
def test_truediv():
    x = AD.Var(3.0)
    y = x / 3.0
    u = AD.Var(1.0, 0.1)
    v = u / x
    z = u / (2 * x)
    assert y._val == 1.0 and y._der == 1.0/3.0, "error with truediv"
    assert v._val == 1.0/3.0 and v._der == -0.7/3.0**2, "error with truediv"
    assert z._val == 1.0/(2*3.0) and z._der == -1.4/36, "error with truediv"
    x = AD.Var([3.0], [1, 0, 0])
    y = AD.Var([1.0], [0, 1, 0])
    w = AD.Var([2.0], [0, 0, 1])
    z = (x + y ** 2 + x * w)/2
    assert np.array_equal(z._val, [5.0]) and np.array_equal(z._der, [1.5, 1.0, 1.5]), "error with truediv"

def test_rtruediv():
    x = AD.Var(3.0)
    y = 3.0 / x
    u = AD.Var(1.0, 2.0)
    v = x / u
    z = (2 * x) / u
    assert y._val == 1.0 and y._der == -1.0/3.0, "error with rtruediv"
    assert v._val == 3.0 and v._der == -5.0, "error with rtruediv"
    assert z._val == 6.0 and z._der == -10.0, "error with rtruediv"
    x = AD.Var(3.0)
    a = AD.Var([ 1., 3, 3, 4.])
    z = 3 / a
    assert np.array_equal(z._val, [3., 1., 1., 0.75]) and np.array_equal(z._der, [-3., -1/3, -1/3, -3/16]), "error with rtruediv"

def test_pow():
    x = AD.Var(2.0)
    y = x**3
    u = AD.Var(3.0, 4.0)
    z = u**2
    v = AD.Var(2.0, 0.5)
    w = v**4
    assert y._val == 8.0 and y._der == 12.0, "error with pow"
    assert z._val == 9.0 and z._der == 24.0, "error with pow"
    assert w._val == 16.0 and w._der == 16.0, "error with pow"
    
def test_eq():
    x = AD.Var(1.0)
    assert x.__eq__(AD.Var(1.0)) == True, "error with eq"
    assert x.__eq__(AD.Var(2.0)) == False, "error with eq"
    x = AD.Var([1, 2], [1, 0, 0])
    y = AD.Var([1, 2], [1, 0, 0])
    with pytest.raises(TypeError) as e:
        x.__eq__(1.0)
    assert str(e.value) == "Cannot compare objects of different types"
    assert x == y, "error with eq"
        
def test_ne():
    x = AD.Var(1.0)
    assert x.__ne__(AD.Var(2.0)) == True, "error with ne"
    assert x.__ne__(AD.Var(1.0)) == False, "error with ne"
    with pytest.raises(TypeError) as e:
        x.__ne__(1.3)
    assert str(e.value) == "Cannot compare objects of different types"
    y = AD.Var([1, 2], [1, 0, 0])
    z = AD.Var([1, 2], [0, 1, 0])
    assert y != z, "error with eq"
"""
#TODO: unfixed
def test_AD_check_tol():
    x = AD.Var(np.pi/4)
    y = AD.tan(x)
    assert y._val == 1.0 and y._der == 2.0, "error with check_tol"
    z = np.linspace(0, 1, 100)
    u = AD.Var(z)
    v = AD.sin(u)
    assert np.array_equal(v._val, np.sin(z)) and np.array_equal(v._der, np.cos(z)), "error with check_tol"

##TODO: wrong caused by check_toi

def test_exp():
    x = AD.Var(1.0, 2.0)
    y = AD.exp(x)
    u = AD.Var(2.0, 3.0)
    z = AD.exp(u) * AD.exp(x)
    assert y._val == np.exp(1.0) and y._der == 2.0*np.exp(1.0), "error with exp"
    assert z._val == np.exp(3.0) and np.abs(z._der - 5.0*np.exp(3.0)) < 10e-8, "error with exp"

def test_log():
    x = AD.Var(2.0, 3.0) 
    y = AD.log(x)
    assert y._val == np.log(x._val) and y._der == 1.5, "error with log"
    with pytest.raises(ValueError) as e:
        u = AD.Var(0)
        z = AD.log(u)
    assert str(e.value) == 'Cannot divide by zero'
    
def test_sin():
    x = AD.Var(np.pi/2)
    y = AD.sin(x)
    u = AD.Var(np.pi/3, 2)
    z = AD.sin(u)
    assert y._val == 1.0 and y._der == 0.0, "error with sin"
    assert np.abs(z._val - np.sin(np.pi/3)) < 10e-8 and np.abs(z._der - 1.0)< 10e-8, "error with sin"

def test_cos():
    x = AD.Var(np.pi)
    y = AD.cos(x)
    u = AD.Var(np.pi/6, 2)
    z = AD.cos(u)
    assert y._val == -1.0 and y._der == 0.0, "error with cos"
    assert np.abs(z._val - np.cos(np.pi/6)) < 10e-8 and np.abs(z._der - (-1.0))< 10e-8, "error with cos"

def test_tan():
    x = AD.Var(np.pi/4)
    y = AD.tan(x)
    assert y._val == 1.0 and y._der == 2.0, "error with tan"
    with pytest.raises(ValueError, match=r"Cannot divide by zero") as e:
        u = AD.Var(np.pi/2)
        z = AD.tan(u)
    assert str(e.value) == 'Cannot divide by zero'
    
def test_arcsin():
    x = AD.Var(0.5)
    y = AD.arcsin(x)
    assert y._val == np.arcsin(0.5) and y._der == 1/np.sqrt(1-0.5**2), "error with arcsin"
    with pytest.raises(ValueError) as e:
        u = AD.Var(1.0)
        v = AD.arcsin(u)
    assert str(e.value) == "Cannot divide by zero"
    
def test_arccos():
    x = AD.Var(0.5)
    y = AD.arccos(x)
    assert y._val == np.arccos(0.5) and y._der == -1/np.sqrt(1-0.5**2), "error with arccos"
    with pytest.raises(ValueError) as e:
        u = AD.Var(1.0)
        v = AD.arccos(u)
    assert str(e.value) == "Cannot divide by zero"
    
def test_arctan():
    x = AD.Var(1.0)
    y = AD.arctan(x)
    assert y._val == np.arctan(1.0) and y._der == 0.5, "error with arctan"

def test_sinh():
    x = AD.Var(1.0)
    y = AD.sinh(x)
    assert y._val == np.sinh(1.0) and y._der == np.cosh(1.0), "error with sinh"

def test_cosh():
    x = AD.Var(1.0)
    y = AD.cosh(x)
    assert y._val == np.cosh(1.0) and y._der == np.sinh(1.0), "error with cosh"
    
def test_tanh():
    x = AD.Var(1.0)
    y = AD.tanh(x)
    assert y._val == np.tanh(1.0) and abs(y._der - 1/np.cosh(1.0)**2) < 1e-8, "error with tanh"
    
def test_sqrt():
    x = AD.Var(4.0)
    y = AD.sqrt(x)
    assert y._val == 2.0 and y._der == -0.25, "error with sqrt"
    with pytest.raises(ValueError) as e:
        u = AD.Var(-1.0)
        v = AD.sqrt(u)
    assert str(e.value) == "Invalid domain"

def test_logistic():
    x = AD.Var(1.0)
    y = AD.logistic(x)
    assert y._val == 1/(1+np.exp(-1.0)) and y._der == np.exp(-1.0)/(1+np.exp(-1.0))**2, "error with logistic"

"""