from scipy import integrate
import numpy as np
from time import time

ti = time()
for i in range(10000):
    x = np.arange(0, 101)
    y = np.arange(0, 101)
    y = np.power(x, 3)
    integ = integrate.simpson(y, x)

print(time()-ti)