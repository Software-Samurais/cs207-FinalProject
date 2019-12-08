import numpy as np

import AD

x = AD.AutoDiff(3.0)

print(x)
print(3+x)
print(x+3)
print(3*x)
print(x/3)
print(x+3)
print(x**2)
print(AD.exp(x**2))

y = AD.AutoDiff(np.pi/2)

print(AD.sin(y))
print(AD.cos(y))


def newton(f, x0, tol=1e-16, max_iter=100):
    """Solves f(x) = 0 using Newton's method.
    
    Args:
    =========
    f (function): Function of interest
    x0 (float): Initial guess
    tol (float): Tolerance value
    max_iter (int): Maximum number of iterations
    
    Returns:
    =========
    xn.val (float): Solution to f(x) = 0 if it exists
                    None if xn.der is zero or if the maximum number of 
                    iterations is reached without satisfying the stopping  
                    criteria
    """
    
    # Initial guess
    xn = x0
    
    for n in range(max_iter):
        
        # Calculate f(xn) and f'(xn) using the AutoDiff class
        fn = f(xn)
        
        # Stop iterating if |f(xn)| is less than the tolerance value and return 
        # the solution, xn
        if abs(fn.val) < tol:
            print(f"Found a solution after {n} iterations.")
            return xn.val
        
        # Check if the derivative is zero
        if fn.der == 0:
            raise ValueError("Encountered zero derivative. No solution.")
            
        # Update guess
        xn = xn - fn.val/fn.der
        
    # Stop iterating if no solution is found within the allowed number of 
    # iterations
    print("Exceeded maximum number of iterations.")
    return None

# Demo Newton's method    
x0 = AD.AutoDiff(1.0)

def f(x):
    return x**2-x-1

print(f"Solution: {newton(f, x0)}")
