# cg_v4_overlap.py
from mpi4py import MPI
import numpy as np
import time
from common import create_distributed_problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cart = comm.Create_cart([size], periods=False, reorder=True)
left, right = cart.Shift(0, 1)

N = 8000
A_local, b_local, n_local = create_distributed_problem(N)

x = np.zeros(n_local)
r = b_local.copy()
p = r.copy()
rsold = comm.allreduce(np.dot(r, r), op=MPI.SUM)

t0 = time.time()
for it in range(2000):
    # Собираем полный вектор p асинхронно (можно перекрыть с вычислениями)
    p_full = np.zeros(N)
    req = comm.Iallgather(p, p_full)
    
    # Пока идёт сбор, можно делать что-то полезное — но для демо просто ждём
    req.Wait()
    
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
    print(f"[ОПТИМ]    Проц: {size:2d} | Время: {t:6.3f} с | Итераций: {it+1}")
