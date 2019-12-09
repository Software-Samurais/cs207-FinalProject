import GuruDiff.forward as forward
import numpy as np
import pytest

def test_init():
    # Simple scalar variables
    x = forward.Var(1.0)
    y = forward.Var(3.0, 1.2)
    assert x.val == 1.0 and x.der == 1.0, "error with init"
    assert y.val == 3.0 and y.der == 1.2, "error with init"
    
    # Scalar variables with array-like inputs
    x = forward.Var([1, 2, 3])
    y = forward.Var([1,2], [2,4])
    assert np.array_equal(x.val, [1, 2, 3]) and np.array_equal(x.der, np.ones(3)), "error with init"
    assert np.array_equal(y.val, [1, 2]) and np.array_equal(y.der, [2, 4]), "error with init"
    
    # Variables for vector functions
    x = forward.Var(1, [1, 0, 0])
    y = forward.Var(2, [0, 1, 0])
    z = forward.Var(3, [0, 0, 1])
    assert x.val == 1.0 and np.array_equal(x.der, [1, 0, 0]), "error with init"
    assert y.val == 2.0 and np.array_equal(y.der, [0, 1, 0]), "error with init"
    assert z.val == 3.0 and np.array_equal(z.der, [0, 0, 1]), "error with init"
    
    # Vector functions
    f = forward.Var([2*x + x*y, -4*x*y])
    g = forward.Var([forward.sin(4*x*y), -4*x*z, 2])
    assert np.array_equal(f.val, [4, -8]) and np.array_equal(f.der, [[4, 1, 0],[-8, -4, 0]]), "error with init"
    assert np.array_equal(g.val, [np.sin(8), -12, 2]) and np.array_equal(g.der, [[8*np.cos(8), 4*np.cos(8), 0], [-12, 0, -4], [0, 0, 0]]), "error with init"
    
def test_val():
    x = forward.Var(1, [1, 0, 0])
    y = forward.Var(2, [0, 1, 0])
    z = forward.Var(3, [0, 0, 1])
    u = forward.Var([x, y])
    f = x + y 
    assert x.val == 1.0, "error with val"
    assert np.array_equal(u.val, [1, 2]), "error with val"
    assert f.val == 3.0, "error with val"
 
def test_der():
    x = forward.Var(1, [1, 0, 0])
    y = forward.Var(2, [0, 1, 0])
    z = forward.Var(3, [0, 0, 1])
    f = x + y + z
    g = 2*x + x*y
    x1 = forward.Var(1.0)
    y1 = forward.Var(1.0, 0.1)
    assert x1.der == 1.0, "error with der"
    assert y1.der == 0.1, "error with der"
    assert np.array_equal(f.der, [1, 1, 1]), "error with der"
    assert np.array_equal(g.der, [4, 1, 0]), "error with der"
    
def test_val_set():
    x = forward.Var(1.0)
    x.val = 2.0
    assert x.val == 2.0, "error with val setter"
    
def test_der_set():
    x = forward.Var(1.0)
    x.der = 2.0
    assert x.der == 2.0, "error with der setter"

def test__repr__():
    
    # Simple scalar variable
    x = forward.Var(1.0)
    assert repr(x) == 'Function value:\n1.0\nDerivative value:\n1.0'
    
    # Scalar variable with array inputs
    x = np.linspace(0, 1, 10)
    y = forward.Var(x)
    assert repr(y) == f"Function values:\n{x}\nDerivative values:\n{np.ones(x.size)}"
    
    # Vector variables
    x = forward.Var([1, 2, 3], [1, 0, 0])
    y = forward.Var([4, 5, 6], [0, 1, 0])
    assert repr(x+y) == f"Function values:\n[5. 7. 9.]\nDerivative values:\n[1. 1. 0.]"

    x = forward.Var(1, [1, 0, 0])
    y = forward.Var(2, [0, 1, 0])
    z = forward.Var(3, [0, 0, 1])
    f = x + y + z
    assert repr(f) == f"Function value:\n6.0\nGradient:\n[1. 1. 1.]"
    g = forward.Var([x, y**2, z**4])
    assert repr(g) == f"Function values:\n{g.val}\nJacobian:\n{g.der}"
    
def test_neg():
    x = forward.Var(1.0, 0.1)
    assert (-x).val == -1, "error with neg"
    assert (-x).der == -0.1, "error with neg"

def test_add():
    x = forward.Var(1.0)
    y = x + 3
    u = forward.Var(3.0, 0.1)
    v = x + u
    assert y._val == 4.0 and y._der == 1.0, "error with add"
    assert v._val == 4.0 and v._der == 1.1, "error with add"
    a = forward.Var(3, [1, 0])
    b = forward.Var(4, [0, 1])
    z = a + b
    assert z._val == 7.0 and np.array_equal(z._der, [1.0, 1.0]), "error with add"

def test_radd():
    x = forward.Var(1.0)
    y = 3 + x
    u = forward.Var(3.0, 0.1)
    v = u + x
    assert y._val == 4.0 and y._der == 1.0, "error with radd"
    assert v._val == 4.0 and v._der == 1.1, "error with radd"
 
def test_sub():
    x = forward.Var(1.0)
    y = x - 3
    u = forward.Var(3.0, 0.1)
    v = x - u
    assert y._val == -2.0 and y._der == 1.0, "error with sub"
    assert v._val == -2.0 and v._der == 0.9, "error with sub"
    a = forward.Var([3], [1, 0])
    b = forward.Var([4], [0, 1])
    z = a - b
    assert np.array_equal(z._val, [-1]) and np.array_equal(z._der, [1.0, -1.0]), "error with add"

def test_rsub():
    x = forward.Var(1.0)
    y = 3 - x
    u = forward.Var(3.0, 0.1)
    v = u - x
    assert y._val == 2.0 and y._der == -1.0, "error with rsub"
    assert v._val == 2.0 and v._der == -0.9, "error with rsub"

def test_mul():
    x = forward.Var(1.0)
    y = 3.0 * x
    u = forward.Var(2.0, 0.1)
    v = x * u
    z = u * (3*x)
    assert y._val == 3.0 and y._der == 3.0, "error with mul"
    assert v._val == 2.0 and v._der == 2.1, "error with mul"
    assert z._val == 6.0 and z._der == 6.3, "error with mul"
    x = forward.Var([3.0], [1, 0])
    y = forward.Var([2], [0, 1])
    z = x * y
    assert np.array_equal(z._val, [6.0]) and np.array_equal(z._der, [2.0, 3.0]), "error with mul"


def test_rmul():
    x = forward.Var(1.0)
    y = x * 3.0
    u = forward.Var(2.0, 0.1)
    v = u * x
    z = (u*3) * x
    assert y._val == 3.0 and y._der == 3.0, "error with rmul"
    assert v._val == 2.0 and v._der == 2.1, "error with rmul"
    assert z._val == 6.0 and z._der == 6.3, "error with rmul"
    x = forward.Var([3.0], [1, 0, 0])
    y = forward.Var([1.0], [0, 1, 0])
    w = forward.Var([2.0], [0, 0, 1])
    z = x + y ** 2 + x * w
    assert np.array_equal(z._val, [10.0]) and np.array_equal(z._der, [3.0, 2.0, 3.0]), "error with rmul"

   
def test_truediv():
    x = forward.Var(3.0)
    y = x / 3.0
    u = forward.Var(1.0, 0.1)
    v = u / x
    z = u / (2 * x)
    assert y._val == 1.0 and y._der == 1.0/3.0, "error with truediv"
    assert v._val == 1.0/3.0 and v._der == -0.7/3.0**2, "error with truediv"
    assert z._val == 1.0/(2*3.0) and z._der == -1.4/36, "error with truediv"
    x = forward.Var([3.0], [1, 0, 0])
    y = forward.Var([1.0], [0, 1, 0])
    w = forward.Var([2.0], [0, 0, 1])
    z = (x + y ** 2 + x * w)/2
    assert np.array_equal(z._val, [5.0]) and np.array_equal(z._der, [1.5, 1.0, 1.5]), "error with truediv"

def test_rtruediv():
    x = forward.Var(3.0)
    y = 3.0 / x
    u = forward.Var(1.0, 2.0)
    v = x / u
    z = (2 * x) / u
    assert y._val == 1.0 and y._der == -1.0/3.0, "error with rtruediv"
    assert v._val == 3.0 and v._der == -5.0, "error with rtruediv"
    assert z._val == 6.0 and z._der == -10.0, "error with rtruediv"
    x = forward.Var(3.0)
    a = forward.Var([ 1., 3, 3, 4.])
    z = 3 / a
    assert np.array_equal(z._val, [3., 1., 1., 0.75]) and np.array_equal(z._der, [-3., -1/3, -1/3, -3/16]), "error with rtruediv"

def test_pow():
    x = forward.Var(2.0)
    y = x**3
    u = forward.Var(3.0, 4.0)
    z = u**2
    v = forward.Var(2.0, 0.5)
    w = v**4
    assert y._val == 8.0 and y._der == 12.0, "error with pow"
    assert z._val == 9.0 and z._der == 24.0, "error with pow"
    assert w._val == 16.0 and w._der == 16.0, "error with pow"
    x = forward.Var(5)
    y = forward.Var(3)
    z = x**y
    assert np.array_equal(z._val, [125]) and np.array_equal(z._der, [75, np.log(5) *(5**3)]), "error with pow"
    
def test_eq():
    x = forward.Var(1.0)
    assert x.__eq__(forward.Var(1.0)) == True, "error with eq"
    assert x.__eq__(forward.Var(2.0)) == False, "error with eq"
    x = forward.Var([1, 2], [1, 0, 0])
    y = forward.Var([1, 2], [1, 0, 0])
    with pytest.raises(TypeError) as e:
        x.__eq__(1.0)
    assert str(e.value) == "Cannot compare objects of different types"
    assert x == y, "error with eq"
        
def test_ne():
    x = forward.Var(1.0)
    assert x.__ne__(forward.Var(2.0)) == True, "error with ne"
    assert x.__ne__(forward.Var(1.0)) == False, "error with ne"
    with pytest.raises(TypeError) as e:
        x.__ne__(1.3)
    assert str(e.value) == "Cannot compare objects of different types"
    y = forward.Var([1, 2], [1, 0, 0])
    z = forward.Var([1, 2], [0, 1, 0])
    assert y != z, "error with eq"

def test_AD_check_tol():
    x = forward.Var(np.pi/4)
    y = forward.tan(x)
    assert y._val == 1.0 and y._der == 2.0, "error with check_tol"
    z = np.linspace(0, 1, 100)
    u = forward.Var(z)
    v = forward.sin(u)
    assert np.array_equal(v._val, np.sin(z)) and np.array_equal(v._der, np.cos(z)), "error with check_tol"

def test_exp():
    x = forward.Var([5.0])
    y = forward.exp(x, 2)
    assert np.array_equal(y._val, [32.]) and np.array_equal(y._der, [np.power(2, 5)*np.log(2)]), "error with exp"

def test_log():
    x = forward.Var(3.0) 
    y = forward.log(x)
    z = forward.log(x, 2)
    assert y.val == np.log(3) and y.der == 1/3, "error with log"
    assert z.val == np.log(3)/np.log(2) and z.der == 1/(3*np.log(2)), "error with log base n"
    with pytest.raises(ValueError) as e:
        u = forward.Var([0])
        z = forward.log(u)
    assert str(e.value) == 'Cannot divide by zero'

  
def test_sin():
    x = forward.Var([np.pi/2])
    y = forward.sin(x)
    u = forward.Var([np.pi/3], [2])
    z = forward.sin(u)
    assert y._val[0] == 1.0 and y._der[0] < 10e-8, "error with sin"
    assert np.abs(z._val[0] - np.sin(np.pi/3)) < 10e-8 and np.abs(z._der[0] - 1.0)< 10e-8, "error with sin"

def test_cos():
    x = forward.Var([np.pi])
    y = forward.cos(x)
    u = forward.Var([np.pi/6], [2])
    z = forward.cos(u)
    assert y._val[0] == -1.0 and y._der < 10e-8, "error with cos"
    assert np.abs(z._val[0] - np.cos(np.pi/6)) < 10e-8 and np.abs(z._der[0] - (-1.0))< 10e-8, "error with cos"

def test_tan():
    x = forward.Var([np.pi/4])
    y = forward.tan(x)
    assert np.abs(y._val[0] - 1.0) <10e-8 and np.abs(y._der[0] - 2.0) < 10e-8, "error with tan"
    with pytest.raises(ValueError, match=r"Cannot divide by zero") as e:
        u = forward.Var([np.pi/2])
        z = forward.tan(u)
    assert str(e.value) == 'Cannot divide by zero'
    
def test_arcsin():
    #x = forward.Var(0.5)
    #y = forward.arcsin(x)
    #assert y._val == np.arcsin(0.5) and y._der == 1/np.sqrt(1-0.5**2), "error with arcsin"
    x = forward.Var([0])
    y = forward.arcsin(x)
    assert y._val[0] == 0 and y._der[0] == 1, "error with arcsin"
    with pytest.raises(ValueError) as e:
        u = forward.Var([1.0])
        v = forward.arcsin(u)
    assert str(e.value) == "x should be in (-1, 1) for arcsin"
    
def test_arccos():
    # x = forward.Var([0.5])
    # y = forward.arccos(x)
    # assert y._val == np.arccos(0.5) and y._der == -1/np.sqrt(1-0.5**2), "error with arccos"
    x = forward.Var([0])
    y = forward.arccos(x)
    assert y._val[0] == np.arccos(0) and y._der[0] == -1, "error with arccos"
    with pytest.raises(ValueError) as e:
        u = forward.Var([-1.0])
        v = forward.arccos(u)
    assert str(e.value) == "x should be in (-1, 1) for arccos"
    
def test_arctan():
    # x = forward.Var(1.0)
    # y = forward.arctan(x)
    # assert y._val == np.arctan(1.0) and y._der == 0.5, "error with arctan"
    x = forward.Var([0])
    y = forward.arctan(x)
    assert y._val[0] == np.arctan(0) and y._der[0] == 1, "error with arccos"
    with pytest.raises(ValueError) as e:
        u = forward.Var([-3.0])
        v = forward.arctan(u)
    assert str(e.value) == "x should be in (-pi/2, pi/2) for arctan"

def test_sinh():
    x = forward.Var([1.0])
    y = forward.sinh(x)
    assert y._val[0] == np.sinh(1.0) and y._der[0] == np.cosh(1.0), "error with sinh"

def test_cosh():
    x = forward.Var([1.0])
    y = forward.cosh(x)
    assert y._val[0] == np.cosh(1.0) and y._der[0] == np.sinh(1.0), "error with cosh"
    
def test_tanh():
    x = forward.Var([1.0])
    y = forward.tanh(x)
    assert y._val[0] == np.tanh(1.0) and abs(y._der[0] - 1/np.cosh(1.0)**2) < 1e-8, "error with tanh"
    
def test_sqrt():
    x = forward.Var([4.0])
    y = forward.sqrt(x)
    assert y._val[0] == 2.0 and y._der[0] == 0.25, "error with sqrt"
    with pytest.raises(ValueError) as e:
        u = forward.Var([-1.0])
        v = forward.sqrt(u)
    assert str(e.value) == "The function value should be greater than zero."

def test_logistic():
    x = forward.Var(1.0)
    y = forward.logistic(x)
    assert y._val == 1/(1+np.exp(-1.0)) and y._der == np.exp(-1.0)/(1+np.exp(-1.0))**2, "error with logistic"
