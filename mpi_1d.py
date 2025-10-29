# mpi_1d.py
from mpi4py import MPI
import numpy as np
import time
from parameters import *
from common import u_init

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Создаём линейную топологию
comm_cart = comm.Create_cart([size], periods=[False])
rank_cart = comm_cart.Get_rank()

def auxiliary_arrays_determination(N, num):
    ave, res = divmod(N, num)
    rcounts = np.array([ave + 1 if k < res else ave for k in range(num)], dtype=np.int32)
    displs = np.zeros(num, dtype=np.int32)
    for k in range(1, num):
        displs[k] = displs[k-1] + rcounts[k-1]
    return rcounts, displs

# Разбиение по x
rcounts_x, displs_x = auxiliary_arrays_determination(N_x + 1, size)
N_x_part = rcounts_x[rank_cart]
N_x_part_aux = N_x_part + (1 if rank_cart > 0 else 0) + (1 if rank_cart < size-1 else 0) + 1  # +1 для границ
u_part = np.empty((M+1, N_x_part_aux, N_y+1), dtype=np.float64)

# Инициализация
for i in range(N_x_part):
    for j in range(N_y+1):
        global_i = displs_x[rank_cart] + i
        u_part[0, i + (1 if rank_cart > 0 else 0), j] = u_init(x[global_i], y[j])

# Граничные условия
for m in range(M+1):
    if rank_cart == 0:
        u_part[m, 0, :] = 0.33
    if rank_cart == size - 1:
        u_part[m, -1, :] = 0.33
    u_part[m, :, 0] = 0.33
    u_part[m, :, -1] = 0.33

start_time = time.time() if rank == 0 else None

for m in range(M):
    # Вычисления (внутренние точки)
    i_start = 1 if rank_cart > 0 else 1
    i_end = N_x_part_aux - 1 if rank_cart < size-1 else N_x_part_aux - 1
    for i in range(i_start, i_end):
        for j in range(1, N_y):
            d2x = (u_part[m, i+1, j] - 2*u_part[m, i, j] + u_part[m, i-1, j]) / h_x**2
            d2y = (u_part[m, i, j+1] - 2*u_part[m, i, j] + u_part[m, i, j-1]) / h_y**2
            d1x = (u_part[m, i+1, j] - u_part[m, i-1, j]) / (2*h_x)
            d1y = (u_part[m, i, j+1] - u_part[m, i, j-1]) / (2*h_y)
            u_part[m+1, i, j] = u_part[m, i, j] + tau * (
                eps * (d2x + d2y) +
                u_part[m, i, j] * (d1x + d1y) +
                u_part[m, i, j]**3
            )

    # Обмен границами
    if rank_cart > 0:
        comm_cart.Sendrecv(
            sendbuf=[u_part[m+1, 1, :], MPI.DOUBLE],
            dest=rank_cart-1, sendtag=0,
            recvbuf=[u_part[m+1, 0, :], MPI.DOUBLE],
            source=rank_cart-1, recvtag=MPI.ANY_TAG
        )
    if rank_cart < size-1:
        comm_cart.Sendrecv(
            sendbuf=[u_part[m+1, -2, :], MPI.DOUBLE],
            dest=rank_cart+1, sendtag=0,
            recvbuf=[u_part[m+1, -1, :], MPI.DOUBLE],
            source=rank_cart+1, recvtag=MPI.ANY_TAG
        )

elapsed = time.time() - start_time if rank == 0 else None
if rank == 0:
    print(f"1D MPI time ({size} procs): {elapsed:.4f} sec")
    np.savez(f"mpi1d_{size}.npz", time=elapsed)
