import numpy as np

class AutoDiff():
    
    def __init__(self, a, da=1.0):
        self.val = a
        self.der = da
        
    def __repr__(self):
        return f"Function value: {self.val}, Derivative value: {self.der}"
        
    def __add__(self, other):
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            return AutoDiff(self.val + other, self.der)
        
    def __radd__(self, other):
        try: 
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            return AutoDiff(self.val + other, self.der)
            
    def __sub__(self, other):
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            return AutoDiff(self.val - other, self.der)
            
    def __rsub__(self, other):
        try:
            return AutoDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            return AutoDiff(other - self.val, self.der)
            
    def __mul__(self, other):
        try:
            return AutoDiff(self.val*other.val, self.der*other.val + self.val*other.der)
        except AttributeError:
            return AutoDiff(other*self.val, other*self.der)
            
    def __rmul__(self, other):
        try:
            return AutoDiff(self.val*other.val, self.der*other.val + self.val*other.der)
        except AttributeError:
            return AutoDiff(other*self.val, other*self.der)
        
    def __truediv__(self, other):
        try: 
            return AutoDiff(self.val/other.val, (self.der*other.val - self.val*other.der)/self.der**2)
        except AttributeError:
            return AutoDiff(self.val/other, self.der/other)
            
    def __pow__(self, n):
        return AutoDiff(self.val**n, n*self.val**(n-1)*self.der)

def check_tol(x, tol=1e-8):
    
    # Check function values
    if abs(x.val - np.round(x.val)) < tol:
        x.val = np.round(x.val)
        
    # Check derivative values
    if abs(x.der - np.round(x.der)) < tol:
        x.der = np.round(x.der)

    return x
        
def sin(x):
    return check_tol(AutoDiff(np.sin(x.val), np.cos(x.val)*x.der))

def cos(x):
    return check_tol(AutoDiff(np.cos(x.val), -np.sin(x.val)*x.der))
    
def tan(x):
    return check_tol(AutoDiff(np.tan(x.val), np.cos(x.val)**(-2)*x.der))
    
def exp(x):
    return check_tol(AutoDiff(np.exp(x.val), np.exp(x.val)*x.der))
