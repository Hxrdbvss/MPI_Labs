# Аналогично full_version, но с полными векторами
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

local_n = rcounts_n[rank]

# Распределение A
A_flat_sizes = [rn * M for rn in rcounts_n]
displs_a = [sum(A_flat_sizes[:i]) for i in range(size)]
A_local_flat = np.zeros(local_n * M)
if size > 1 or rank == 0:
    comm.Scatterv([A_data, A_flat_sizes, displs_a, MPI.DOUBLE], A_local_flat, root=0)
A_local = A_local_flat.reshape(local_n, M) if local_n > 0 else np.empty((0, M))

# b_full на всех
b_local = np.zeros(local_n)
if size > 1 or rank == 0:
    comm.Scatterv([b_data, rcounts_n, displs_n, MPI.DOUBLE], b_local, root=0)
b = np.zeros(N)
comm.Allgatherv(b_local, [b, rcounts_n, displs_n, MPI.DOUBLE])

# Инициализация
x = np.zeros(M)
r = b.copy()

# s = A.T @ r
contrib = A_local.T @ r[displs_n[rank]:displs_n[rank] + local_n] if local_n > 0 else np.zeros(M)
s = np.zeros(M)
comm.Allreduce(contrib, s, op=MPI.SUM)

gamma = np.dot(s, s)

p = s.copy()

# Замер
start_time = time.time()

max_iters = M
tol = 1e-6

iter_count = 0
for k in range(max_iters):
    iter_count += 1
    if np.sqrt(gamma) < tol:
        break

    # q = A @ p
    local_q = A_local @ p if local_n > 0 else np.empty(0)
    q = np.zeros(N)
    comm.Allgatherv(local_q, [q, rcounts_n, displs_n, MPI.DOUBLE])

    qq = np.dot(q, q)
    if qq == 0:
        break
    alpha = gamma / qq

    x += alpha * p
    r -= alpha * q

    # s_new = A.T @ r
    contrib = A_local.T @ r[displs_n[rank]:displs_n[rank] + local_n] if local_n > 0 else np.zeros(M)
    s = np.zeros(M)
    comm.Allreduce(contrib, s, op=MPI.SUM)

    gamma_new = np.dot(s, s)
    if np.sqrt(gamma_new) < tol:
        break

    beta = gamma_new / gamma
    gamma = gamma_new
    p = s + beta * p

end_time = time.time()
comp_time = end_time - start_time

if rank == 0:
    print(f"Время вычисления (ядро): {comp_time:.4f} сек")
    print(f"Итераций: {iter_count}")

    # Верификация
    A = A_data.reshape(N, M)
    b = b_data
    x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
    max_diff = np.max(np.abs(x - x_numpy))
    print(f"Макс. разница с numpy: {max_diff:.2e}")
