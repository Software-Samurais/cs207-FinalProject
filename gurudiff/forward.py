import numpy as np

class Var:
    """Forward mode variable"""
    
    def __init__(self, a, da=None):
        """Returns a forward mode variable.
        
        Parameters:
        ===========
        a (int, float, or array): Function value(s)
        da (int, float, array, or None): Derivative value(s); by default, set to None
        
        Attributes:
        ===========
        self.val : Function value(s)
        self.der : Derivative value(s)
        
        Returns:
        ========
        Var: Forward mode variable
        """
        # Set function value(s)
        self._val = np.asarray(a)
        
        # Length of the input 
        n = self._val.size
        
        # Entries of a can be instances of the Var class when defining vector
        # valued function variables. 
        if isinstance(self._val.any(), Var):
            
            vals = []
            jac = []
            
            # Iterate over the entries in self.val and self.der to extract the 
            # relevant information (i.e. function and derivative values).
            for element in self._val:
                try:
                    vals.append(element._val)
                    jac.append(element._der)
                except AttributeError:
                    vals.append(element)
                    jac.append(np.zeros(n))
            
            # Reset attributes
            self._val = np.asarray(vals).astype(float).flatten()
            self._der = np.asarray(jac).astype(float)
            
        else:
            self._val = self._val.astype(float)
            
            # Set derivative value if none specified by the user
            if da is None:
                if n == 1:
                    self._der = np.asarray(1.)
                else:
                    self._der = np.ones(n)
            else: 
                self._der = np.asarray(da).astype(float)
                
    def __repr__(self):
        """Returns relevant information about a forward mode variable."""
        # Simple scalar variables
        if self._val.size == 1 and self._der.size == 1:
            return f"Function value:\n{self._val}\nDerivative value:\n{self._der}"
        
        # Scalar variables with array-like inputs
        if self._val.size == self._der.size:
            return f"Function values:\n{self._val}\nDerivative values:\n{self._der}"
        
        # Vector function variables
        if self._val.shape != self._der.shape:
            try:
                n, m = self._der.shape
                return f"Function values:\n{self._val}\nJacobian:\n{self._der}"
            except ValueError:
                return f"Function value:\n{float(self._val)}\nGradient:\n{self._der}"
    
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
    
    # ====================
    # Operator overloading
    # ====================
    def __neg__(self):
        """Returns the negative of a forward mode variable.
        
        The negative of a forward mode variable is defined as a new forward mode
        variable with the input's attributes multiplied by a negative. 
        """
        return Var(-self._val, -self._der)
    
    def __add__(self, other):
        """Returns the sum of two forward mode variables.
        
        The sum of two forward mode variables is defined as a new forward mode 
        variable whose attributes are the sum of the input attributes.
        """
        try:
            return Var(self._val + other._val, self._der + other._der)
        
        # Handles cases where other is not an instance of Var.
        except AttributeError:
            return Var(self._val + other, self._der)
        
    def __radd__(self, other):
        """Returns the sum of two forward mode variables."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Returns the difference between two forward mode variables.
        
        The difference between two forward mode variables is defined as a new 
        forward mode variable whose attributes are the difference of the input
        attributes.
        """
        return self.__add__(-other)
    
    def __rsub__(self, other):
        """Returns the difference between two forward mode variables."""
        return -self.__sub__(other)
    
    def __mul__(self, other):
        """Returns the product of two forward mode variables.
        
        The product of two forward mode variable is defined as a new forward 
        mode variable whose function value is the product of the input values 
        and whose derivative value is calculated according to the product rule 
        of differentiation.
        """
        try:
            return Var(self._val*other._val, self._val*other._der + self._der*other._val)
        
        # Handles cases where other is not an instance of Var.
        except AttributeError:
            return Var(other*self._val, other*self._der)
    
    def __rmul__(self, other):
        """Returns the product of two forward mode variables."""
        return self.__mul__(other)
        
    def __truediv__(self, other):
        """Returns the quotient of two forward mode variables.
        
        The quotient of two forward mode variables is defined as a new forward 
        mode variable whose function value is the quotient of the input values 
        and whose derivative value is calculated according to the quotient rule 
        of differentiation.
        """
        try: 
            return Var(self._val/other._val, (self._der*other._val - self._val*other._der)/other._val**2)
        
        # Handles cases where other is not an instance of Var.
        except AttributeError:
            return Var(self._val/other, self._der/other)
            
    def __rtruediv__(self, other):
        """Returns the quotient of two forward mode variables."""
        try: 
            return Var(other._val/self._val, (other._der*self._val - other._val*self._der)/self._val**2)
        
        # Handles cases where other is not an instance of Var.
        except AttributeError:
            return Var(other/self._val, -other*self._der/self._val**2)
    
    def __pow__(self, p):
        """Returns a new forward mode variable raised to some power, p."""
        try:
            return Var([self._val**p._val], [p._val*self._val**(p._val-1) * self._der, np.log(self._val) * self._val ** p._val * p._der])
        except AttributeError:
            return Var(self._val ** p, p * self._val ** (p - 1) * self._der)

    # ====================
    # Comparison operators
    # ====================
    
    def __eq__(self, other):
        """Checks for equality of two forward mode variables.
        
        Two forward mode variables are equal if both their function and 
        derivative values are equal. 
        """
        try:
            if np.array_equal(self.val, other.val) and np.array_equal(self.der, other.der):
                return True
            else:
                return False
        
        # Raises an exception when comparing a forward mode variable to an 
        # object of a different type (e.g. float, np.ndarray).
        except AttributeError:
            raise TypeError("Cannot compare objects of different types")
            
    def __ne__(self, other):
        """Checks that two forward mode variables are not equal."""
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
    
    Parameters:
    ===========
    - x (Var): Scalar forward mode variable
    
    Returns:
    ========
    - x (Var): Updated scalar forward mode variable, if the difference 
      between the actual value and the rounded value of the attributes is less 
      than some tolerance; otherwise, the input is returned
    """
    
    # Check function value(s)
    try:
        if abs(x._val - np.round(x._val)) < tol:
            x._val = np.round(x._val)
    
    except ValueError:
        with np.nditer(x._val, op_flags=["readwrite"]) as it:
            for entry in it:
                if abs(entry - np.round(entry)) < tol:
                    entry[...] = np.round(entry)
            
    # Check derivative value(s)
    try:
        if abs(x._der - np.round(x._der)) < tol:
            x._der = np.round(x._der)
        
    except ValueError:
        with np.nditer(x._der, op_flags=["readwrite"]) as it:
            for entry in it:
                if abs(entry - np.round(entry)) < tol:
                    entry[...] = np.round(entry)
                
    return x

# Trigonometric functions
# =======================

def sin(x):
    """Returns the sine of a forward mode variable and its derivative.
    
    Parameters:
    ===========
    - x (Var): Forward mode variable
    
    Returns:
    ========
    - (Var): Sine value and the corresponding derivative value; `check_tol`
      is called to remove rounding errors
    """
    return check_tol(Var(np.sin(x._val), np.cos(x._val)*x._der))

def cos(x):
    """Returns the cosine of a forward mode variable and its derivative.
    
    Parameters:
    =========== 
    - x (Var): Forward mode variable
    
    Returns:
    ========
    - (Var): Cosine value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    return check_tol(Var(np.cos(x._val), -np.sin(x._val)*x._der))
    
def tan(x):
    """Returns the tangent of a forward mode variable and its derivative.
    
    Parameters:
    =========== 
    - x (Var): Forward mode variable
    
    Returns:
    ========
    - (Var): Tangent value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if (type(x._val) is float or type(x._val) is int and np.abs(x._val - np.pi/2) > 10e-8) or np.abs(x._val - np.pi/2).all() > 10e-8:
        return check_tol(Var(np.tan(x._val), np.cos(x._val)**(-2)*x._der))
    else:
        raise ValueError("Cannot divide by zero")

# Inverse trigonometric functions
# ===============================
     
def arcsin(x):
    """Returns the inverse sine of a forward mode variable."""
    if ((x._val - 1) < 0).all() and ((x._val + 1) >0).all():
        return check_tol(Var(np.arcsin(x._val), x._der/np.sqrt(1-x._val**2)))
    else:
        raise ValueError("x should be in (-1, 1) for arcsin")
        
def arccos(x):
    """Returns the inverse cosine of a forward mode variable."""
    if ((x._val - 1) < 0).all() and ((x._val + 1) > 0).all():
        return check_tol(Var(np.arccos(x._val), -x._der/np.sqrt(1-x._val**2)))
    else:
        raise ValueError("x should be in (-1, 1) for arccos")
        
def arctan(x):
    """Returns the inverse tangent of a forward mode variable."""
    if ((x._val - np.pi/2) < 0).all() and ((x._val + np.pi/2) > 0).all():
        return check_tol(Var(np.arctan(x._val), x._der/(1+x._val**2)))
    else:
        raise ValueError("x should be in (-pi/2, pi/2) for arctan")
    
# Exponentials
# ============
# NOTE: The natural base is treated as a special case. All others are handled 
# via operator overloading of the power method. 

def exp(x, base=None):
    """Returns the exponential of a forward mode variable and its derivative.
    
    Parameters:
    =========== 
    - x (Var): Forward mode variable
    
    Returns:
    ========
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
    """Returns the logarithm of a forward mode variable and its derivative.
    
    Parameters:
    =========== 
    - x (Var): Forward mode variable
    
    Returns:
    ========
    - (Var): Logarithm value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if (type(x._val) is float or type(x._val) is int and x._val != 0) or x._val.all():
        if base is None:
            return check_tol(Var(np.log(x._val), x._der/x._val))
        else:
            return check_tol(Var(np.log(x._val)/np.log(base), 1/(x._val*np.log(base))))
    else:
        raise ValueError("Cannot divide by zero")
        
        
# Hyperbolic functions
# ====================

def sinh(x):
    """Returns the hyperbolic sine of a forward mode variable."""
    return (exp(x) - exp(-x))/2
    
def cosh(x):
    """Returns the hyperbolic cosine of a forward mode variable."""
    return (exp(x) + exp(-x))/2
    
def tanh(x):
    """Returns the hyperbolic tangent of a forward mode variable."""
    return sinh(x)/cosh(x)
    
# Square root
# ===========

def sqrt(x):
    """Square root of a forward mode variable."""
    if (x._val > 0).all():
        return check_tol(Var(np.sqrt(x._val), x._der/(2*np.sqrt(x._val))))
    else:
        raise ValueError("The function value should be greater than zero.")
        
# Logisitic function
# =================

def logistic(x):
    """Standard logistic function.
    
    Parameters:
    ===========
    x (Var): Forward mode variable
    
    Returns:
    ========
    Var: Standard logistic function with x as the argument
    """
    return 1/(1 + exp(-x))

