# common.py
import numpy as np
from parameters import eps

def u_init(x, y):
    return 0.5 * np.tanh(1/eps * ((x-0.5)**2 + (y-0.5)**2 - 0.35**2)) - 0.17
