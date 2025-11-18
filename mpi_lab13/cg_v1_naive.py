# cg_v1_naive.py
from mpi4py import MPI
import numpy as np
import time
from common import create_distributed_problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 8000
A_local, b_local, n_local = create_distributed_problem(N)

# Все векторы — локальные!
x = np.zeros(n_local)
r = b_local.copy()
p = r.copy()
rsold = comm.allreduce(np.dot(r, r), op=MPI.SUM)

t0 = time.time()
for it in range(2000):
    # SpMV: A_local (n_local × N) @ p_global → но p_global собирается из p_local!
    # Поэтому сначала собираем полный p на каждом процессе через Allgatherv
    p_full = np.zeros(N)
    comm.Allgather(p, p_full)
    
    Ap_local = A_local @ p_full

    pAp = comm.allreduce(np.dot(p, Ap_local), op=MPI.SUM)
    alpha = rsold / (pAp + 1e-30)
    
    x += alpha * p
    r -= alpha * Ap_local

    rsnew = comm.allreduce(np.dot(r, r), op=MPI.SUM)
    if np.sqrt(rsnew) < 1e-8:
        break

    p = r + (rsnew / (rsold + 1e-30)) * p
    rsold = rsnew

t = time.time() - t0
if rank == 0:
    print(f"[НАИВНАЯ]  Проц: {size:2d} | Время: {t:6.3f} с | Итераций: {it+1}")
