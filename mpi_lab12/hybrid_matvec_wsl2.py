from mpi4py import MPI
import numpy as np
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 4000

if rank == 0:
    print("=== Hybrid MPI + OpenMP/NumPy ===")
    print("MPI processes:", size)
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
    print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
    print()

if rank == 0:
    A = np.random.random((N, N)).astype(np.float64)
    x = np.random.random(N).astype(np.float64)
else:
    A = None
    x = None

x = comm.bcast(x, root=0)

rows = N // size
start = rank * rows
end = (rank + 1) * rows

if rank == 0:
    A_local = A[start:end, :]
else:
    A_local = np.empty((rows, N), dtype=np.float64)

comm.Scatter([A, MPI.DOUBLE], [A_local, MPI.DOUBLE], root=0)

t0 = time.time()
y_local = A_local.dot(x)
t_local = time.time() - t0

y = None
if rank == 0:
    y = np.empty(N, dtype=np.float64)

comm.Gather(y_local, y, root=0)

t_max = comm.reduce(t_local, op=MPI.MAX, root=0)

if rank == 0:
    print("Largest per-process time:", t_max)
    print("y[0:5] =", y[:5])

