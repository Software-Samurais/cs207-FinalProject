import numpy as np

class AutoDiff():
    """Returns a scalar variable for forward mode automatic differentiation.
    
    Attributes:
    - _val (float): Function value (private)
    - _der (float): Derivative value (private)
    
    NOTE: There are getter and setter methods that correspond to `_val` and 
    `_der` which allow the user to easily access and manually update the 
    function and derivative values of an AutoDiff instance.
    """
    
    def __init__(self, a, da=1.0):
        """Defines function and derivative values of a scalar forward mode 
        variable.
        
        Args:
        - a (float): Function value
        - da (float): Derivative value; defaults to 1.0 if the user does not 
          pass in a value
          
        Returns:
        - None
        """
        self._val = a
        self._der = da
    
    @property
    def val(self):
        """Returns function value of a scalar forward mode variable."""
        return self._val
        
    @property
    def der(self):
        """Returns derivative value of a scalar forward mode variable."""
        return self._der
        
    @val.setter
    def val(self, a):
        self._val = a
        
    @der.setter
    def der(self, da):
        self._der = da
        
    def __repr__(self):
        """Returns relevant information about a scalar forward mode variable."""
        return f"Function value: {self._val}, Derivative value: {self._der}"

    def  __neg__(self):
        """Overloads negation."""
        return AutoDiff(-self._val, -self._der)
        
    def __add__(self, other):
        """Overloads addition.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the sum of the 
          arguments. If a float is passed as the second argument, the derivative
          value of the new scalar forward mode variable will be the same as that
          of the first argument.
        """
        try:
            return AutoDiff(self._val + other._val, self._der + other._der)
        except AttributeError:
            return AutoDiff(self._val + other, self._der)
        
    def __radd__(self, other):
        """Overloads addition from the right.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the sum of the 
          arguments. If a float is passed as the second argument, the derivative
          value of the new scalar forward mode variable will be the same as that
          of the first argument.
        """
        return self.__add__(other)
            
    def __sub__(self, other):
        """Overloads subtraction.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the difference of
          the arguments. If a float is passed as the second argument, the 
          derivative value of the new scalar forward mode variable will be the 
          same as that of the first argument.
        """
        return self.__add__(-other)
            
    def __rsub__(self, other):
        """Overloads subtraction from the right.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the difference of
          the arguments. If a float is passed as the second argument, the 
          derivative value of the new scalar forward mode variable will be the 
          same as that of the first argument.
        """
        return -self.__sub__(other)
            
    def __mul__(self, other):
        """Overloads multiplication.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the product of
          the arguments. This operation follows the product and chain rules of 
          differentiation. Passing in a float as the second argument corresponds
          to scaling the function and derivative values of the first argument.
        """
        try:
            return AutoDiff(self._val*other._val, self._der*other._val + self._val*other._der)
        except AttributeError:
            return AutoDiff(other*self._val, other*self._der)
            
    def __rmul__(self, other):
        """Overloads multiplication from the right.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the product of
          the arguments. This operation follows the product and chain rules of 
          differentiation. Passing in a float as the second argument corresponds
          to scaling the function and derivative values of the first argument.
        """
        return self.__mul__(other)
        
    def __truediv__(self, other):
        """Overloads division.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the quotient of
          the arguments. This operation follows the quotient and chain rules of 
          differentiation. Passing in a float as the second argument corresponds
          to scaling the function and derivative values of the first argument.
        """
        try: 
            return AutoDiff(self._val/other._val, (self._der*other._val - self._val*other._der)/other._val**2)
        except AttributeError:
            return AutoDiff(self._val/other, self._der/other)
    
    def __rtruediv__(self, other):
        """Overloads division from the right.
        
        Args:
        - self (AutoDiff): Scalar forward mode variable
        - other (AutoDiff or float): Accepts a scalar forward mode variable or 
          an ordinary float
          
        Returns:
        - (AutoDiff): New scalar forward mode variable that is the quotient of
          the arguments. This operation follows the quotient and chain rules of 
          differentiation. Passing in a float as the second argument corresponds
          to scaling the function and derivative values of the first argument.
        """
        try: 
            return AutoDiff(self._val/other._val, (other._der*self._val - other._val*self._der)/self._val**2)
        except AttributeError:
            return AutoDiff(other/self._val, -other/self._val **2)

    def __pow__(self, n):
        """Overloads the power operator.
        
        Args:
        - n (int or float): Exponent
        
        Returns:
        - (AutoDiff): New scalar forward mode variable raised to the n power;  
          This operation follows the power and chain rules of differentiation. 
        """
        return AutoDiff(self._val**n, n*(self._val**(n-1))*self._der)

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
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - x (AutoDiff): Updated scalar forward mode variable, if the difference 
      between the actual value and the rounded value is less than some 
      tolerance; otherwise, the input is returned
    """
    # Check function value
    if abs(x._val - np.round(x._val)) < tol:
        x._val = np.round(x._val)
        
    # Check derivative value
    if abs(x._der - np.round(x._der)) < tol:
        x._der = np.round(x._der)

    return x

def sin(x):
    """Returns the sine of a scalar forward mode variable and its derivative.
    
    Args:
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - (AutoDiff): Sine value and the corresponding derivative value; `check_tol`
      is called to remove rounding errors
    """
    return check_tol(AutoDiff(np.sin(x._val), np.cos(x._val)*x._der))

def cos(x):
    """Returns the cosine of a scalar forward mode variable and its derivative.
    
    Args: 
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - (AutoDiff): Cosine value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    return check_tol(AutoDiff(np.cos(x._val), -np.sin(x._val)*x._der))
    
def tan(x):
    """Returns the tangent of a scalar forward mode variable and its derivative.
    
    Args: 
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - (AutoDiff): Tangent value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if cos(x)._val != 0:
        return check_tol(AutoDiff(np.tan(x._val), np.cos(x._val)**(-2)*x._der))
    else:
        raise ValueError("Cannot divide by zero")
    
def exp(x):
    """Returns the exponential of a forward mode variable and its derivative.
    
    Args: 
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - (AutoDiff): Exponential value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    return check_tol(AutoDiff(np.exp(x._val), np.exp(x._val)*x._der))
    
def log(x):
    """Returns the logarithm of a forward mode variable and its derivative.
    
    Args: 
    - x (AutoDiff): Scalar forward mode variable
    
    Returns:
    - (AutoDiff): Logarithm value and the corresponding derivative value; 
      `check_tol` is called to remove rounding errors
    """
    if x._val != 0:
        return check_tol(AutoDiff(np.log(x._val), x._der/x._val))
    else:
        raise ValueError("Cannot divide by zero")
