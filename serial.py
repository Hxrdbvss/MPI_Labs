# serial.py
import numpy as np
import time
from parameters import *
from common import u_init

def u_boundary(y, t):  # одинаковые для всех границ
    return 0.33

# Инициализация
u = np.empty((M+1, N_x+1, N_y+1))
for i in range(N_x+1):
    for j in range(N_y+1):
        u[0, i, j] = u_init(x[i], y[j])

# Граничные условия
for m in range(M+1):
    u[m, 0, :] = u_boundary(y, t[m])
    u[m, -1, :] = u_boundary(y, t[m])
    u[m, :, 0] = u_boundary(x, t[m])
    u[m, :, -1] = u_boundary(x, t[m])

start_time = time.time()

for m in range(M):
    for i in range(1, N_x):
        for j in range(1, N_y):
            d2x = (u[m, i+1, j] - 2*u[m, i, j] + u[m, i-1, j]) / h_x**2
            d2y = (u[m, i, j+1] - 2*u[m, i, j] + u[m, i, j-1]) / h_y**2
            d1x = (u[m, i+1, j] - u[m, i-1, j]) / (2*h_x)
            d1y = (u[m, i, j+1] - u[m, i, j-1]) / (2*h_y)
            u[m+1, i, j] = u[m, i, j] + tau * (
                eps * (d2x + d2y) +
                u[m, i, j] * (d1x + d1y) +
                u[m, i, j]**3
            )

elapsed = time.time() - start_time
print(f"Serial time: {elapsed:.4f} sec")

# Сохраняем результат (опционально)
np.savez("serial_result", u=u, x=x, y=y, t=t, time=elapsed)
