# parameters.py
import numpy as np

# Параметры из задания
a, b, c, d = -2, 2, -2, 2
t_0, T = 0, 4
eps = 10**(-1.5)

# Тестовые параметры для отчёта
N_x = N_y = 100
M = 1000

x = np.linspace(a, b, N_x + 1)
y = np.linspace(c, d, N_y + 1)
h_x = x[1] - x[0]
h_y = y[1] - y[0]
t = np.linspace(t_0, T, M + 1)
tau = t[1] - t[0]
