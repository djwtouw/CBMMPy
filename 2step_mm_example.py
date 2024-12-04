import numpy as np
import matplotlib.pyplot as plt


# %%

def foo(x):
    return -np.sin(x) + 0.1 * x**2 + 0.1 * np.exp(x**5 / 5)


def d_foo(x):
    return -np.cos(x) + 0.2 * x + 0.1 * np.exp(x**5 / 5) * x**4


# Set range of x
x = np.linspace(0, 1.605, 1000)

# Compute y
y = foo(x)

# Supporting points
x0 = 0.3
x1 = 0.6
x2 = 0.8

# Compute polynomial coefficients for the first majorization function
a = -d_foo(x0) / (x1 - x0) / 2
b = d_foo(x0) - 2 * a * x0
c = foo(x0) - a * x0**2 - b * x0

# Compute first majorization function
y1 = a * x**2 + b * x + c

# Compute polynomial coefficients for the first majorization function
a = -d_foo(x0) / (x2 - x0) / 2
b = d_foo(x0) - 2 * a * x0
c = foo(x0) - a * x0**2 - b * x0

# Compute first majorization function
y2 = a * x**2 + b * x + c

# Get true minimum
x3 = x[np.where(y == min(y))][0]

# Plot
plt.figure(dpi=500)
plt.plot(x, y)
plt.plot(x[y1 <= max(y)], y1[y1 <= max(y)])
plt.plot(x[y2 <= max(y)], y2[y2 <= max(y)])

plt.xticks([x0, x1, x2, x3])
plt.tight_layout()
plt.show()
