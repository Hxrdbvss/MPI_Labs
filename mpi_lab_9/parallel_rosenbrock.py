# parallel_rosenbrock.py
# ПОЛНОСТЬЮ ИСПРАВЛЕНО: размеры, Sendrecv, сборка

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm_cart = comm.Create_cart([size], periods=[False], reorder=True)

# ==============================
# Параметры
# ==============================
a, b = 0.0, 1.0
t_0, T = 0.0, 2.0
eps = 10**(-1.5)
N = 10000       # 10001 точек
M = 2500        # 2501 шагов
alpha = 0.5

h = (b - a) / N
tau = (T - t_0) / M
t = np.linspace(t_0, T, M + 1)

def u_left(t_val): return np.sin(np.pi * t_val)
def u_right(t_val): return 0.0
def u_init(x): return np.sin(np.pi * x)

# ==============================
# Распределение
# ==============================
if rank == 0:
    ave, res = divmod(N + 1, size)
    rcounts = np.array([ave + 1 if k < res else ave for k in range(size)], dtype=np.int32)
    displs = np.zeros(size, dtype=np.int32)
    for k in range(1, size):
        displs[k] = displs[k-1] + rcounts[k-1]
else:
    rcounts = np.empty(size, dtype=np.int32)
    displs = np.empty(size, dtype=np.int32)

comm.Bcast(rcounts, root=0)
comm.Bcast(displs, root=0)

N_part = rcounts[rank]
N_start = displs[rank]

# Расширенные блоки
N_part_aux = N_part + 2
if rank == 0: N_part_aux = N_part + 1
if rank == size - 1: N_part_aux = N_part + 1

u_local = np.zeros(N_part_aux)
u_local_next = np.zeros(N_part_aux)

# Инициализация
x_local = np.linspace(a + N_start * h, a + (N_start + N_part) * h, N_part)
u_init_local = u_init(x_local)

if rank == 0:
    u_local[:N_part] = u_init_local
    u_local[0] = u_left(t_0)
else:
    u_local[1:N_part+1] = u_init_local
if rank == size - 1:
    u_local[-1] = u_right(t_0)

# ==============================
# Локальные функции
# ==============================
def f_local(y, t_val, h, N_loc, left_val, right_val, eps):
    if N_loc <= 2: return np.array([])
    f_loc = np.zeros(N_loc - 2)
    for i in range(1, N_loc-1):
        l = left_val if i == 1 else y[i-2]
        c = np.clip(y[i-1], -1.5, 1.5)
        r = right_val if i == N_loc-2 else y[i]
        f_loc[i-1] = eps * (r - 2*c + l) / h**2 + c * (r - l) / (2*h) + c**3
    return f_loc

def diagonal_local(y, t_val, h, N_loc, left_val, right_val, eps, tau, alpha):
    if N_loc <= 2: return np.array([]), np.array([]), np.array([])
    a = np.zeros(N_loc-2)
    b = np.zeros(N_loc-2)
    c = np.zeros(N_loc-2)
    for i in range(1, N_loc-1):
        idx = i - 1
        if i == 1:
            a[idx] = -alpha*tau*(eps/h**2 - y[0]/(2*h))
            b[idx] = 1. - alpha*tau*(-2*eps/h**2 + (y[1] - left_val)/(2*h) + 3*y[0]**2)
            c[idx] = -alpha*tau*(eps/h**2 + y[0]/(2*h))
        elif i == N_loc-2:
            a[idx] = -alpha*tau*(eps/h**2 - y[-1]/(2*h))
            b[idx] = 1. - alpha*tau*(-2*eps/h**2 + (right_val - y[-2])/(2*h) + 3*y[-1]**2)
        else:
            a[idx] = -alpha*tau*(eps/h**2 - y[idx]/(2*h))
            b[idx] = 1. - alpha*tau*(-2*eps/h**2 + (y[idx+1] - y[idx-1])/(2*h) + 3*y[idx]**2)
            c[idx] = -alpha*tau*(eps/h**2 + y[idx]/(2*h))
    return a, b, c

def thomas(a, b, c, d):
    n = len(d)
    if n == 0: return np.array([])
    x = np.zeros(n)
    for i in range(1, n):
        factor = a[i] / b[i-1]
        b[i] -= factor * c[i-1]
        d[i] -= factor * d[i-1]
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

# ==============================
# Основной цикл
# ==============================
start_time = MPI.Wtime() if rank == 0 else 0.0

for m in range(M):
    left_val = u_left(t[m] + tau/2) if rank == 0 else u_local[0]
    right_val = u_right(t[m] + tau/2) if rank == size - 1 else u_local[-1]

    rhs = f_local(u_local, t[m] + tau/2, h, N_part_aux, left_val, right_val, eps)
    a_d, b_d, c_d = diagonal_local(u_local, t[m], h, N_part_aux,
                                   u_left(t[m]), u_right(t[m]), eps, tau, alpha)
    w1 = thomas(a_d, b_d, c_d, rhs)

    # Обновляем ТОЛЬКО внутренние точки
    inner_start = 1 if rank > 0 else 1
    inner_end = N_part_aux - 1 if rank < size - 1 else N_part_aux - 1
    u_local_next[inner_start:inner_end] = u_local[inner_start:inner_end] + tau * w1.real
    u_local_next = np.clip(u_local_next, -2.0, 2.0)

    # Граничные условия
    if rank == 0:
        u_local_next[0] = u_left(t[m+1])
    if rank == size - 1:
        u_local_next[-1] = u_right(t[m+1])

    # Обмен: передаём МАССИВЫ, а не скаляры
    if size > 1:
        left_src, right_dst = comm_cart.Shift(0, 1)
        if rank > 0:
            comm_cart.Sendrecv(
                [np.array([u_local_next[1]]), 1, MPI.DOUBLE],
                dest=left_src, sendtag=0,
                recvbuf=[np.array([u_local_next[0]]), 1, MPI.DOUBLE],
                source=left_src, recvtag=0
            )
        if rank < size - 1:
            comm_cart.Sendrecv(
                [np.array([u_local_next[-2]]), 1, MPI.DOUBLE],
                dest=right_dst, sendtag=0,
                recvbuf=[np.array([u_local_next[-1]]), 1, MPI.DOUBLE],
                source=right_dst, recvtag=0
            )

    u_local, u_local_next = u_local_next.copy(), u_local

elapsed = MPI.Wtime() - start_time if rank == 0 else 0.0

# ==============================
# Вывод времени
# ==============================
if rank == 0:
    print(f"Parallel time ({size} procs): {elapsed:.4f} sec")

# ==============================
# Сбор решения
# ==============================
if rank == 0:
    u_full = np.zeros((M+1, N+1))
    u_full[0, :] = u_init(np.linspace(a, b, N+1))

for m in range(1, M+1):
    data = u_local[1:N_part_aux-1] if rank > 0 and rank < size-1 else u_local[1:-1]
    if rank == 0:
        data = u_local[1:N_part]
    if rank == size - 1:
        data = u_local[1:-1]
    if rank == 0:
        u_full[m, N_start:N_start + len(data)] = data
    else:
        comm.Send(data, dest=0, tag=m)

if rank == 0:
    np.save(f"u_parallel_{size}procs.npy", u_full)
    plt.figure(figsize=(10,6))
    plt.imshow(u_full.T, extent=[t_0, T, a, b], aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='u(x,t)')
    plt.title(f'Parallel ROS1 ({size} procs, {elapsed:.2f}s)')
    plt.savefig(f"parallel_solution_{size}.png", dpi=150)
    plt.close()
