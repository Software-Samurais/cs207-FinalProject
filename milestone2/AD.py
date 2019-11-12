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
        
    def __add__(self, other):
        try:
            return AutoDiff(self._val + other._val, self._der + other._der)
        except AttributeError:
            return AutoDiff(self._val + other, self._der)
        
    def __radd__(self, other):
        try: 
            return AutoDiff(self._val + other._val, self._der + other._der)
        except AttributeError:
            return AutoDiff(self._val + other, self._der)
            
    def __sub__(self, other):
        try:
            return AutoDiff(self._val - other._val, self._der - other._der)
        except AttributeError:
            return AutoDiff(self._val - other, self._der)
            
    def __rsub__(self, other):
        try:
            return AutoDiff(other._val - self._val, other._der - self._der)
        except AttributeError:
            return AutoDiff(other - self._val, self._der)
            
    def __mul__(self, other):
        try:
            return AutoDiff(self._val*other._val, self._der*other._val + self._val*other._der)
        except AttributeError:
            return AutoDiff(other*self._val, other*self._der)
            
    def __rmul__(self, other):
        try:
            return AutoDiff(self._val*other._val, self._der*other._val + self._val*other._der)
        except AttributeError:
            return AutoDiff(other*self._val, other*self._der)
        
    def __truediv__(self, other):
        try: 
            return AutoDiff(self._val/other._val, (self._der*other._val - self._val*other._der)/self._der**2)
        except AttributeError:
            return AutoDiff(self._val/other, self._der/other)
            
    def __pow__(self, n):
        return AutoDiff(self._val**n, n*self._val**(n-1)*self._der)

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
    return check_tol(AutoDiff(np.tan(x._val), np.cos(x._val)**(-2)*x._der))
    
def exp(x):
    return check_tol(AutoDiff(np.exp(x._val), np.exp(x._val)*x._der))
