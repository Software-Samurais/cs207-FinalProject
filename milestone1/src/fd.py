# Import libraries
import matplotlib.pyplot as plt
import numpy as np

def numerical_diff(f, h):
    
    def df(x):
        
        # Calculate the derivative using the limit definition
        derivative = (f(x + h) - f(x))/h
        
        return derivative
    
    return df

# Function of interest
def f(x):
    return np.log(x)

# Analytic derivative
def dfdx(x):
    return 1/x

# Define the h values
h_values = [1e-1, 1e-7, 1e-15]

# Domain of interest
x = np.linspace(0.2, 0.4, 100)

# Analytic derivative on the domain of interest
true_df = dfdx(x)

# Preallocate an array of zeros to store numerical derivatives
n = len(x)
dfdx_num = np.zeros((n, n))

# Loop over the h values
for h in h_values:
    
    k = h_values.index(h)
    
    dfdx_h = numerical_diff(f, h)
    
    dfdx_num[:, k] = dfdx_h(x)

# Write the output to a file
with open("fd_out.dat", "w+") as f:
    
    # File header
    f.write("# Forward difference method\n")
    f.write("# x, true, h=1e-1, h=1e-7, h=1e-15\n")
      
    # Results
    for k in range(n):
        f.write("{} {} {} {} {}\n".format(x[k], true_df[k], dfdx_num[k,0], dfdx_num[k,1], dfdx_num[k,2]))
