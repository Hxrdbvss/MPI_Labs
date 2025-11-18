# lab5_2d_FINAL_FIXED.py
import sys
import numpy as np
from mpi4py import MPI
import math

def auxiliary_arrays_determination(total_size, num_procs):
    block_size = total_size // num_procs
    remainder = total_size % num_procs
    rcounts = [block_size] * num_procs
    for i in range(remainder):
        rcounts[i] += 1
    displs = [0] * num_procs
    for i in range(1, num_procs):
        displs[i] = displs[i-1] + rcounts[i-1]
    return rcounts, displs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Проверка квадрата ---
sqrt_size = int(math.sqrt(size))
if sqrt_size * sqrt_size != size:
    if rank == 0:
        print("ОШИБКА: число процессов не квадрат!", file=sys.stderr)
    sys.exit(1)

num_row = num_col = sqrt_size
comm_row = comm.Split(rank // num_col, rank)
comm_col = comm.Split(rank % num_col, rank)
row_rank = comm_row.Get_rank()
col_rank = comm_col.Get_rank()

# --- Чтение M, N ---
if rank == 0:
    with open('in.dat') as f:
        M, N = map(int, f.readline().split())
    start_time = MPI.Wtime()
else:
    M, N = 0, 0
    start_time = 0.0

M, N = comm.bcast((M, N), root=0)

# --- Размеры блоков ---
rcounts_M, displs_M = [0]*num_row, [0]*num_row
rcounts_N, displs_N = [0]*num_col, [0]*num_col
if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)

rcounts_M = comm.bcast(rcounts_M, root=0)
displs_M = comm.bcast(displs_M, root=0)
rcounts_N = comm.bcast(rcounts_N, root=0)
displs_N = comm.bcast(displs_N, root=0)

M_part = rcounts_M[row_rank]
N_part = rcounts_N[col_rank]

# --- Распределение A ---
A_part = np.zeros((M_part, N_part), dtype=np.float64)
if rank == 0:
    A_full = np.loadtxt('AData.dat').reshape(M, N)
    for i in range(num_row):
        for j in range(num_col):
            trank = i * num_col + j
            block = A_full[displs_M[i]:displs_M[i]+rcounts_M[i],
                          displs_N[j]:displs_N[j]+rcounts_N[j]]
            if trank == 0:
                A_part[:] = block
            else:
                comm.send(block, dest=trank, tag=1)
else:
    A_part = comm.recv(source=0, tag=1)

# --- Распределение x ---
x_part = np.zeros(N_part, dtype=np.float64)
if rank == 0:
    x_full = np.loadtxt('xData.dat')
    for j in range(num_col):
        x_block = x_full[displs_N[j]:displs_N[j]+rcounts_N[j]]
        for i in range(num_row):
            trank = i * num_col + j
            if trank == 0:
                x_part[:] = x_block
            else:
                comm.send(x_block, dest=trank, tag=2)
else:
    x_part = comm.recv(source=0, tag=2)
x_part = comm_col.bcast(x_part, root=0)

# --- Локальное умножение ---
b_part_local = np.dot(A_part, x_part)  # (M_part,)

# --- СБОР ПО СТРОКЕ: Gatherv (без Reduce!) ---
if row_rank == 0:
    recvcounts = np.array(rcounts_M, dtype=np.int32)
    displs = np.zeros(num_row, dtype=np.int32)
    for i in range(1, num_row):
        displs[i] = displs[i-1] + recvcounts[i-1]
    
    b_row_full = np.zeros(M, dtype=np.float64)

    comm_row.Gatherv(
        sendbuf=b_part_local,
        recvbuf=[b_row_full, recvcounts, displs, MPI.DOUBLE],
        root=0
    )

    # Копируем свою часть
    start = displs[row_rank]
    end = start + rcounts_M[row_rank]
    b_part = b_row_full[start:end].copy()

else:
    comm_row.Gatherv(sendbuf=b_part_local, recvbuf=None, root=0)
    b_part = None

# --- СБОР НА rank 0 ---
if row_rank == 0:
    if rank == 0:
        b_full = np.zeros(M, dtype=np.float64)
        for i in range(num_row):
            src = i * num_col
            start = displs_M[i]
            if src == 0:
                b_full[start:start + len(b_part)] = b_part
            else:
                data = comm.recv(source=src, tag=3)
                b_full[start:start + len(data)] = data
        end_time = MPI.Wtime()
        print(f"Время: {end_time - start_time:.6f} сек")
        print(f"||b|| = {np.linalg.norm(b_full):.6f}")
    else:
        comm.send(b_part, dest=0, tag=3)

MPI.Finalize()
