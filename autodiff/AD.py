import numpy as np

class AutoDiff():
    
    def __init__(self, a, da=1.0):
        self._val = a
        self._der = da
    
    @property
    def val(self):
        return self._val
        
    @property
    def der(self):
        return self._der
        
    def __repr__(self):
        return f"Function value: {self._val}, Derivative value: {self._der}"

    def  __neg__(self):
        return AutoDiff(-self._val, -self._der)
        
    def __add__(self, other):
        try:
            return AutoDiff(self._val + other._val, self._der + other._der)
        except AttributeError:
            return AutoDiff(self._val + other, self._der)
        
    def __radd__(self, other):
        return self.__add__(other)
            
    def __sub__(self, other):
        return self.__add__(-other)
            
    def __rsub__(self, other):
         return -self.__sub__(other)
            
    def __mul__(self, other):
        try:
            return AutoDiff(self._val*other._val, self._der*other._val + self._val*other._der)
        except AttributeError:
            return AutoDiff(other*self._val, other*self._der)
            
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        try: 
            return AutoDiff(self._val/other._val, (self._der*other._val - self._val*other._der)/other._val**2)
        except AttributeError:
            return AutoDiff(self._val/other, self._der/other)
    
    def __rtruediv__(self, other):
        try: 
            return AutoDiff(self._val/other._val, (other._der*self._val - other._val*self._der)/self._val**2)
        except AttributeError:
            return AutoDiff(other/self._val, -other/self._val **2)

    def __pow__(self, n):
        return AutoDiff(self._val**n, n*(self._val**(n-1))*self._der)

def check_tol(x, tol=1e-8):
    
    # Check function values
    if abs(x._val - np.round(x._val)) < tol:
        x._val = np.round(x._val)
        
    # Check derivative values
    if abs(x._der - np.round(x._der)) < tol:
        x._der = np.round(x._der)

    return x

def sin(x):
    return check_tol(AutoDiff(np.sin(x._val), np.cos(x._val)*x._der))

def cos(x):
    return check_tol(AutoDiff(np.cos(x._val), -np.sin(x._val)*x._der))
    
def tan(x):
    if cos(x)._val != 0:
        return check_tol(AutoDiff(np.tan(x._val), np.cos(x._val)**(-2)*x._der))
    else:
        raise ValueError("Cannot divide by zero")
    
def exp(x):
    return check_tol(AutoDiff(np.exp(x._val), np.exp(x._val)*x._der))
    
def log(x):
    if x._val != 0:
        return check_tol(AutoDiff(np.log(x._val), x._der/x._val))
    else:
        raise ValueError("Cannot divide by zero")
