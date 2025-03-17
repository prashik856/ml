# Create an array
import numpy as np

w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])

# One way of creating model function
f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b

# Using for loop
f_for = 0
for i in range(0, w.size):
    f_for = f_for + w[i] * x[i]
f_for = f_for + b

# Using numpy dot product
f_numpy = np.dot(w, x) + b

