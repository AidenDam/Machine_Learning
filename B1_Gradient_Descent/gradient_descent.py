import math
import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return 2*x + 5*np.cos(x)

def F(x):
    return x**2 + 5*np.sin(x)

def myGD(eta, x_init, step = 100):
    x = [x_init]
    for it in range(step):
        x_new = x[-1] - eta*f(x[-1])
        if abs(f(x[-1])) < 1e-3:
            break
        x.append(x_new)
    return x

x1 = myGD(0.1, -5)
x2 = myGD(0.1, 5)

print('x1 =', x1[-1])
print('x2 =', x2[-1])

x = np.linspace(-6, 6)
plt.plot(x, [F(it) for it in x])

plt.show()