# serial_rosenbrock.py
import numpy as np
from numpy import empty, linspace, float64
import matplotlib.pyplot as plt
import time

# Параметры
a, b = 0.0, 1.0
t_0, T = 0.0, 2.0
eps = 10**(-1.5)
N = 10000
M = 2500
alpha = 0.5

h = (b - a) / N
tau = (T - t_0) / M
x = linspace(a, b, N + 1)
t = linspace(t_0, T, M + 1)

# Граничные и начальные условия
def u_left(t): return np.sin(np.pi * t)
def u_right(t): return 0.0
def u_init(x): return np.sin(np.pi * x)

# Правая часть
def f(y, t_val, h, N, u_left, u_right, eps):
    f = empty(N - 1, dtype=float64)
    f[0] = eps * (y[1] - 2*y[0] + u_left(t_val)) / h**2 \
           + y[0] * (y[1] - u_left(t_val)) / (2*h) + y[0]**3
    for n in range(1, N-2):
        f[n] = eps * (y[n+1] - 2*y[n] + y[n-1]) / h**2 \
               + y[n] * (y[n+1] - y[n-1]) / (2*h) + y[n]**3
    f[N-2] = eps * (u_right(t_val) - 2*y[N-2] + y[N-3]) / h**2 \
             + y[N-2] * (u_right(t_val) - y[N-3]) / (2*h) + y[N-2]**3
    return f

# Формирование диагоналей
def diagonal_preparation(y, t_val, h, N, u_left, u_right, eps, tau, alpha):
    a = empty(N-1); b = empty(N-1); c = empty(N-1)
    b[0] = 1. - alpha*tau*(-2*eps/h**2 + (y[1] - u_left(t_val))/(2*h) + 3*y[0]**2)
    c[0] = -alpha * tau * (eps/h**2 + y[0]/(2*h))
    for n in range(1, N-2):
        a[n] = -alpha*tau*(eps/h**2 - y[n]/(2*h))
        b[n] = 1. - alpha*tau*(-2*eps/h**2 + (y[n+1] - y[n-1])/(2*h) + 3*y[n]**2)
        c[n] = -alpha*tau*(eps/h**2 + y[n]/(2*h))
    a[N-2] = -alpha*tau*(eps/h**2 - y[N-2]/(2*h))
    b[N-2] = 1. - alpha*tau*(-2*eps/h**2 + (u_right(t_val) - y[N-3])/(2*h) + 3*y[N-2]**2)
    return a, b, c

# Метод прогонки
def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N = len(d)
    x = empty(N, dtype=float64)
    for n in range(1, N):
        coef = a[n] / b[n-1]
        b[n] -= coef * c[n-1]
        d[n] -= coef * d[n-1]
    x[N-1] = d[N-1] / b[N-1]
    for n in range(N-2, -1, -1):
        x[n] = (d[n] - c[n] * x[n+1]) / b[n]
    return x

# Основной цикл
def solve_serial():
    y = u_init(x)[1:-1]
    u_history = [np.copy(np.concatenate(([u_left(t_0)], y, [u_right(t_0)])))]

    start_time = time.time()
    for m in range(M):
        rhs = f(y, t[m] + tau/2, h, N, u_left, u_right, eps)
        a_diag, b_diag, c_diag = diagonal_preparation(y, t[m], h, N, u_left, u_right, eps, tau, alpha)
        w1 = consecutive_tridiagonal_matrix_algorithm(a_diag, b_diag, c_diag, rhs)
        y = y + tau * w1.real

        u_full = np.concatenate(([u_left(t[m+1])], y, [u_right(t[m+1])]))
        u_history.append(u_full.copy())

    elapsed = time.time() - start_time
    return np.array(u_history), elapsed

if __name__ == "__main__":
    u_sol, t_serial = solve_serial()
    print(f"Serial time: {t_serial:.4f} sec")

    # Сохранение результата
    np.save("u_serial.npy", u_sol)
    plt.figure(figsize=(10,6))
    plt.imshow(u_sol.T, extent=[t_0, T, a, b], aspect='auto', origin='lower')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('t'); plt.ylabel('x')
    plt.title('Serial ROS1 Solution')
    plt.savefig("serial_solution.png")
    plt.close()
