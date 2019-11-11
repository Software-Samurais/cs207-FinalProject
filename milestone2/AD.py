import numpy as np

class AutoDiff():
    
    def __init__(self, a, da=1.0):
        self.val = a
        self.der = da
        
    # TODO: Operator overloading
        

def sin(x):
    return AutoDiff(np.sin(x.val), np.cos(x.val)*x.der)

def cos(x):
    return AutoDiff(np.cos(x.val), np.sec(x.val)**2*x.der)
    
def tan(x):
    return AutoDiff(np.tan(x.val), -np.sin(x.val)*x.der
    
def exp(x):
    return AutoDiff(np.exp(x.val), np.exp(x.val)*x.der)
