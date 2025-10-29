# mpi_2d.py
from mpi4py import MPI
import numpy as np
import time
from parameters import *
from common import u_init

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Проверяем, что size — квадрат
import math
num_row = num_col = int(math.sqrt(size))
if num_row * num_col != size:
    if rank == 0:
        print("Ошибка: число процессов должно быть квадратом!")
    exit()

comm_cart = comm.Create_cart([num_row, num_col], periods=[False, False], reorder=True)
rank_cart = comm_cart.Get_rank()
my_row, my_col = comm_cart.Get_coords(rank_cart)

def auxiliary_arrays_determination(N, num):
    ave, res = divmod(N, num)
    rcounts = np.array([ave + 1 if k < res else ave for k in range(num)], dtype=np.int32)
    displs = np.zeros(num, dtype=np.int32)
    for k in range(1, num):
        displs[k] = displs[k-1] + rcounts[k-1]
    return rcounts, displs

rcounts_x, displs_x = auxiliary_arrays_determination(N_x + 1, num_col)
rcounts_y, displs_y = auxiliary_arrays_determination(N_y + 1, num_row)

N_x_part = rcounts_x[my_col]
N_y_part = rcounts_y[my_row]

N_x_aux = N_x_part + (1 if my_col > 0 else 0) + (1 if my_col < num_col-1 else 0)
N_y_aux = N_y_part + (1 if my_row > 0 else 0) + (1 if my_row < num_row-1 else 0)

u_part = np.empty((M+1, N_x_aux, N_y_aux), dtype=np.float64)

# Инициализация
for i in range(N_x_part):
    for j in range(N_y_part):
        gx = displs_x[my_col] + i
        gy = displs_y[my_row] + j
        u_part[0, i + (1 if my_col > 0 else 0), j + (1 if my_row > 0 else 0)] = u_init(x[gx], y[gy])

# Граничные условия
for m in range(M+1):
    if my_col == 0: u_part[m, 0, :] = 0.33
    if my_col == num_col-1: u_part[m, -1, :] = 0.33
    if my_row == 0: u_part[m, :, 0] = 0.33
    if my_row == num_row-1: u_part[m, :, -1] = 0.33

start_time = time.time() if rank == 0 else None

for m in range(M):
    # Вычисления
    i_start = 1 if my_col > 0 else 1
    i_end = N_x_aux - 1 if my_col < num_col-1 else N_x_aux - 1
    j_start = 1 if my_row > 0 else 1
    j_end = N_y_aux - 1 if my_row < num_row-1 else N_y_aux - 1

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            d2x = (u_part[m, i+1, j] - 2*u_part[m, i, j] + u_part[m, i-1, j]) / h_x**2
            d2y = (u_part[m, i, j+1] - 2*u_part[m, i, j] + u_part[m, i, j-1]) / h_y**2
            d1x = (u_part[m, i+1, j] - u_part[m, i-1, j]) / (2*h_x)
            d1y = (u_part[m, i, j+1] - u_part[m, i, j-1]) / (2*h_y)
            u_part[m+1, i, j] = u_part[m, i, j] + tau * (
                eps * (d2x + d2y) +
                u_part[m, i, j] * (d1x + d1y) +
                u_part[m, i, j]**3
            )

    # Обмен по X (горизонталь)
    if my_col > 0:
        comm_cart.Sendrecv(
            sendbuf=[u_part[m+1, 1, :], MPI.DOUBLE],
            dest=comm_cart.Get_cart_rank([my_row, my_col-1]),
            recvbuf=[u_part[m+1, 0, :], MPI.DOUBLE],
            source=comm_cart.Get_cart_rank([my_row, my_col-1])
        )
    if my_col < num_col-1:
        comm_cart.Sendrecv(
            sendbuf=[u_part[m+1, -2, :], MPI.DOUBLE],
            dest=comm_cart.Get_cart_rank([my_row, my_col+1]),
            recvbuf=[u_part[m+1, -1, :], MPI.DOUBLE],
            source=comm_cart.Get_cart_rank([my_row, my_col+1])
        )

    # Обмен по Y (вертикаль)
    if my_row > 0:
        temp_send = u_part[m+1, :, 1].copy()
        temp_recv = np.empty(N_x_aux, dtype=np.float64)
        comm_cart.Sendrecv(
            sendbuf=[temp_send, MPI.DOUBLE],
            dest=comm_cart.Get_cart_rank([my_row-1, my_col]),
            recvbuf=[temp_recv, MPI.DOUBLE],
            source=comm_cart.Get_cart_rank([my_row-1, my_col])
        )
        u_part[m+1, :, 0] = temp_recv
    if my_row < num_row-1:
        temp_send = u_part[m+1, :, -2].copy()
        temp_recv = np.empty(N_x_aux, dtype=np.float64)
        comm_cart.Sendrecv(
            sendbuf=[temp_send, MPI.DOUBLE],
            dest=comm_cart.Get_cart_rank([my_row+1, my_col]),
            recvbuf=[temp_recv, MPI.DOUBLE],
            source=comm_cart.Get_cart_rank([my_row+1, my_col])
        )
        u_part[m+1, :, -1] = temp_recv

elapsed = time.time() - start_time if rank == 0 else None
if rank == 0:
    print(f"2D MPI time ({size} procs): {elapsed:.4f} sec")
    np.savez(f"mpi2d_{size}.npz", time=elapsed)
