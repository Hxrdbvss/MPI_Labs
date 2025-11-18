import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_counts(n, size):
    base = n // size
    extra = n % size
    rcounts = [base + 1 if i < extra else base for i in range(size)]
    displs = [sum(rcounts[:i]) for i in range(size)]
    return rcounts, displs

if rank == 0:
    with open('in.dat', 'r') as f:
        line = f.read().strip()
        N, M = map(int, line.split())
    A_data = np.loadtxt('AData.dat')
    b_data = np.loadtxt('bData.dat')
else:
    N = None
    M = None
    A_data = None
    b_data = None

N = comm.bcast(N, root=0)
M = comm.bcast(M, root=0)

rcounts_n, displs_n = compute_counts(N, size)
rcounts_m, displs_m = compute_counts(M, size)

local_n = rcounts_n[rank]
local_m = rcounts_m[rank]

# Распределение A
A_flat_sizes = [rn * M for rn in rcounts_n]
displs_a = [sum(A_flat_sizes[:i]) for i in range(size)]
A_local_flat = np.zeros(local_n * M)
if size > 1 or rank == 0:  # Для 1 процесса без Scatterv
    comm.Scatterv([A_data, A_flat_sizes, displs_a, MPI.DOUBLE], A_local_flat, root=0)
A_local = A_local_flat.reshape(local_n, M) if local_n > 0 else np.empty((0, M))

# Распределение b
b_local = np.zeros(local_n)
if size > 1 or rank == 0:
    comm.Scatterv([b_data, rcounts_n, displs_n, MPI.DOUBLE], b_local, root=0)

# Инициализация x_local = 0, r_local = b_local
x_local = np.zeros(local_m)
r_local = b_local.copy() if local_n > 0 else np.empty(0)

# s_local = A.T @ r (распределённо)
if local_n > 0:
    contrib = A_local.T @ r_local
else:
    contrib = np.zeros(M)
s_full = np.zeros(M)
comm.Allreduce(contrib, s_full, op=MPI.SUM)
# Распределить s_full на s_local
s_local = np.zeros(local_m)
comm.Scatterv([s_full, rcounts_m, displs_m, MPI.DOUBLE], s_local, root=0)

local_dot = np.dot(s_local, s_local)
gamma = comm.allreduce(local_dot, op=MPI.SUM)

p_local = s_local.copy()

# Замер времени цикла
start_time = time.time()

max_iters = M
tol = 1e-6  # Ранняя остановка (доп. задание)

iter_count = 0
for k in range(max_iters):
    iter_count += 1
    if np.sqrt(gamma) < tol:
        break

    # Сбор p_full
    p_full = np.zeros(M)
    comm.Allgatherv(p_local, [p_full, rcounts_m, displs_m, MPI.DOUBLE])

    # q_local = A_local @ p_full
    q_local = A_local @ p_full if local_n > 0 else np.empty(0)

    # alpha = gamma / (q @ q)
    local_qq = np.dot(q_local, q_local) if local_n > 0 else 0
    qq = comm.allreduce(local_qq, op=MPI.SUM)
    if qq == 0:
        break
    alpha = gamma / qq

    # Обновление x и r
    x_local += alpha * p_local
    if local_n > 0:
        r_local -= alpha * q_local

    # s_new = A.T @ r (Allreduce)
    contrib = A_local.T @ r_local if local_n > 0 else np.zeros(M)
    s_new_full = np.zeros(M)
    comm.Allreduce(contrib, s_new_full, op=MPI.SUM)
    # Распределить s_new_full на s_local
    s_local = np.zeros(local_m)
    comm.Scatterv([s_new_full, rcounts_m, displs_m, MPI.DOUBLE], s_local, root=0)

    # gamma_new
    local_dot = np.dot(s_local, s_local)
    gamma_new = comm.allreduce(local_dot, op=MPI.SUM)
    if np.sqrt(gamma_new) < tol:
        break

    beta = gamma_new / gamma
    gamma = gamma_new
    p_local = s_local + beta * p_local

end_time = time.time()
comp_time = end_time - start_time

# Сбор x_full на rank 0
x_full = np.zeros(M) if rank == 0 else None
comm.Gatherv(x_local, [x_full, rcounts_m, displs_m, MPI.DOUBLE], root=0)

if rank == 0:
    print(f"Время вычисления (ядро): {comp_time:.4f} сек")
    print(f"Итераций: {iter_count}")

    # Верификация
    A = A_data.reshape(N, M)
    b = b_data
    x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
    max_diff = np.max(np.abs(x_full - x_numpy))
    print(f"Макс. разница с numpy: {max_diff:.2e}")
