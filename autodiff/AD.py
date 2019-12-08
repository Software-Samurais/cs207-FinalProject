import numpy as np

class Var:
    
    def __init__(self, a, da=None):
        #Old __init__ method
        """
        self._val = np.asarray(a).astype(float)
        if da is None:     
            if self._val.size == 1:
                self._der = 1.0
            else:
                self._der = np.ones(self._val.size)
        else:
            self._der = np.asarray(da).astype(float)
        """
        # New constructor
        # NOTE: This makes many of the unit tests fail!
         # Simple scalar variables
        if type(a) is float or type(a) is int:
            self._val = float(a)
            if da is None:
                self._der = 1.0
            elif type(da) is float or type(da) is int:
                self._der = float(da)
            else:
                raise KeyError("The format of value and derivative is not align")
                
        # Variables with array-like inputs
        if type(a) is list or type(a) is np.ndarray:
            self._val = np.asarray(a)
            
            # Vector functions
            # NOTE: This works well for most cases, except those where a 
            # component is independent of x, y, or z.
            if isinstance(self._val.any(), Var):
                
                vals = []
                jac = []
                
                for element in self._val:
                    try:
                        vals.append(element._val)
                        jac.append(element._der)
                    except AttributeError:
                        vals.append(element)
                        jac.append(np.zeros(len(a)))
                
                self._val = np.asarray(vals).astype(float).flatten()
                self._der = np.asarray(jac).astype(float)
                
            else:
                if da is None:     
                    self._der = np.ones(self._val.size)
                else:
                    self._der = np.asarray(da).astype(float)
       
        
    def __repr__(self):
        try:
            # For vector functions
            if self._val.shape != self._der.shape:
                try:
                    n, m = self._der.shape
                    return f"Function values:\n{self._val}\nJacobian:\n{self._der}"
                except ValueError:
                    return f"Function values:\n{self._val}\nGradient:\n{self._der}"
            
            # For scalar functions using arrays
            else:
                return f"Function values:\n{self._val}\nDerivative values:\n{self._der}"
        
        # Simple scalar variables
        except AttributeError:
            return f"Function value:\n{self._val}\nDerivative value:\n{self._der}"            
    
    @property
    def val(self):
        return self._val
    
    @property
    def der(self):
        return self._der
    
    @val.setter
    def val(self, a):
        self._val = a
        
    @der.setter
    def der(self, da):
        self._der = da
    
    # Operator overloading
    # ====================
    def __neg__(self):
        return Var(-self._val, -self._der)
    
    def __add__(self, other):
        try:
            return Var(self._val + other._val, self._der + other._der)
        except AttributeError:
            return Var(self._val + other, self._der)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return -self.__sub__(other)
    
    def __mul__(self, other):
        try:
            return Var(self._val*other._val, self._val*other._der + self._der*other._val)
        except AttributeError:
            return Var(other*self._val, other*self._der)
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        try: 
            return Var(self._val/other._val, (self._der*other._val - self._val*other._der)/other._val**2)
        except AttributeError:
            return Var(self._val/other, self._der/other)
            
    def __rtruediv__(self, other):
        try: 
            return Var(other._val/self._val, (other._der*self._val - other._val*self._der)/self._val**2)
        except AttributeError:
            return Var(other/self._val, -other*self._der/self._val**2)
    
    def __pow__(self, p):
        try:
            return Var([self._val**p._val], [p._val*self._val**(p._val-1) * self._der, np.log(self._val) * self._val ** p._val * p._der])
        except AttributeError:
            return Var(self._val ** p, p * self._val ** (p - 1) * self._der)

            
    # Comparison operators
    # ====================
    
    def __eq__(self, other):
        try:
            if isinstance(self.val, float) and isinstance(other.val, float):
                if self.val == other.val and self.der == other.der:
                    return True
                else:
                    return False
            else:
                if np.array_equal(self.val, other.val) and np.array_equal(self.der, other.der):
                    return True
                else:
                    return False
        except AttributeError:
            raise TypeError("Cannot compare objects of different types")
            
    def __ne__(self, other):
        return not self.__eq__(other)

def check_tol(x, tol=1e-8):
    """Returns rounded function and/or derivative values.
    
    Although some elementary functions have exact values when evaluated at 
    certain points (e.g. sin(pi/2) or log(1)), rounding error can affect all 
    calculations because some constants cannot be represented to exact precision
    (e.g. pi) and because floating point operations (e.g. addition, subtraction,
    multiplication, etc.) incur rounding error. The `check_tol` method is used 
    to check the difference between the calculated values and their rounded 
    counterparts. If this difference falls below a certain tolerance (e.g. 1e-8 
    by default), the rounded value will replace the calculated value.
    
    Args:
    - x (Var): Scalar Var mode variable
    
    Returns:
    - x (Var): Updated scalar Var mode variable, if the difference 
      between the actual value and the rounded value is less than some 
      tolerance; otherwise, the input is returned
    """
    try:
        # Check function value
        if abs(x._val - np.round(x._val)) < tol:
            x._val = np.round(x._val)
            
        # Check derivative value
        if abs(x._der - np.round(x._der)) < tol:
            x._der = np.round(x._der)
    except ValueError:
        for k in range(x._val.size):
            try:
                if abs(x._val[k] - np.round(x._val[k])) < tol:
                    x._val[k] = np.round(x._val[k])
            except IndexError:
                pass
            
            try:
                if abs(x._der[k] - np.round(x._der[k])) < tol:
                    x._der[k] = np.round(x._der[k])
            except IndexError:
                pass

    return x

# Trigonometric functions
# =======================

def sin(x):
    """Returns the sine of a scalar Var mode variable and its derivative.
    
    Args:
    - x (Var): Scalar Var mode variable
    
    Returns:
    - (Var): Sine value and the corresponding derivative value; `check_tol`
      is called to remove rounding errors
    """
    return Var(np.sin(x._val), np.cos(x._val)*x._der)

def cos(x):
    """Returns the cosine of a scalar Var mode variable and its derivative.
    
    Args: 
    - x (Var): Scalar Var mode variable
    
    Returns:
    - (Var): Cosine value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    return Var(np.cos(x._val), -np.sin(x._val)*x._der)
    
def tan(x):
    """Returns the tangent of a scalar Var mode variable and its derivative.
    
    Args: 
    - x (Var): Scalar Var mode variable
    
    Returns:
    - (Var): Tangent value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if (type(x._val) is float or type(x._val) is int and np.abs(x._val - np.pi/2) > 10e-8) or np.abs(x._val - np.pi/2).all() > 10e-8:
        return Var(np.tan(x._val), np.cos(x._val)**(-2)*x._der)
    else:
        raise ValueError("Cannot divide by zero")

# Inverse trigonometric functions
# ===============================
     
def arcsin(x):
    if ((x._val - 1) < 0).all() and ((x._val + 1) >0).all():
        return Var(np.arcsin(x._val), x._der/np.sqrt(1-x._val**2))
    else:
        raise ValueError("x should be in (-1, 1) for arcsin")
        
def arccos(x):
    if ((x._val - 1) < 0).all() and ((x._val + 1) > 0).all():
        return Var(np.arccos(x._val), -x._der/np.sqrt(1-x._val**2))
    else:
        raise ValueError("x should be in (-1, 1) for arccos")
        
def arctan(x):
    if ((x._val - np.pi/2) < 0).all() and ((x._val + np.pi/2) > 0).all():
        return Var(np.arctan(x._val), x._der/(1+x._val**2))
    else:
        raise ValueError("x should be in (-pi/2, pi/2) for arctan")
    
# Exponentials
# ============
# NOTE: The natural base is treated as a special case. All others are handled 
# via operator overloading of the power method. 

def exp(x, base=None):
    """Returns the exponential of a Var mode variable and its derivative.
    
    Args: 
    - x (Var): Scalar Var mode variable
    
    Returns:
    - (Var): Exponential value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if base is None:
        return Var(np.exp(x._val), np.exp(x._val)*x._der)
    else:
        return Var(np.power(base, x._val), np.power(base, x._val)*np.log(base))

# Logarithms
# ==========
 
def log(x, base=None):
    """Returns the logarithm of a Var mode variable and its derivative.
    
    Args: 
    - x (Var): Scalar Var mode variable
    
    Returns:
    - (Var): Logarithm value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if (type(x._val) is float or type(x._val) is int and x._val != 0) or x._val.all():
        if base is None:
            return Var(np.log(x._val), x._der/x._val)
        else:
            return Var(np.log(x._val)/np.log(base), 1/(x._val*np.log(base)))
    else:
        raise ValueError("Cannot divide by zero")
        
        
# Hyperbolic functions
# ====================

def sinh(x):
    return (exp(x) - exp(-x))/2
    
def cosh(x):
    return (exp(x) + exp(-x))/2
    
def tanh(x):
    return sinh(x)/cosh(x)
    
# Square root
# ===========

def sqrt(x):
    if (x._val > 0).all():
        return Var(np.sqrt(x._val), x._der/(2*np.sqrt(x._val)))
    else:
        raise ValueError("x should be larger than 0")
        
# Logisitc function
# =================

def logistic(x):
    return 1/(1 + exp(-x))

