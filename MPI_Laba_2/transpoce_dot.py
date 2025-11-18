from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = time.time()
    with open('in.dat', 'r') as f:
        M, N = map(int, f.read().split())
    local_Ms = [M // size + (1 if i < M % size else 0) for i in range(size)]
    rcounts_A = [lm * N for lm in local_Ms]
    displs_A = [0] + np.cumsum(rcounts_A[:-1]).tolist()
    rcounts_x = local_Ms
    displs_x = [0] + np.cumsum(rcounts_x[:-1]).tolist()
    A = np.loadtxt('AData.dat').reshape(M, N)
    x = np.loadtxt('xData.dat')
    b_seq = np.dot(A.T, x)
    print(f"Sequential dot: {b_seq[0]} ... {b_seq[-1]} (first and last element for verification)")
else:
    M = None
    N = None
    local_Ms = None
    rcounts_A = None
    displs_A = None
    rcounts_x = None
    displs_x = None
    A = None
    x = None

M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)
local_M = comm.scatter(local_Ms, root=0)

A_part = np.empty(local_M * N, dtype=np.float64)
comm.Scatterv([A, rcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
A_part = A_part.reshape(local_M, N)

x_part = np.empty(local_M, dtype=np.float64)
comm.Scatterv([x, rcounts_x, displs_x, MPI.DOUBLE], x_part, root=0)

b_temp = np.dot(A_part.T, x_part)

if rank == 0:
    b = np.zeros(N, dtype=np.float64)
else:
    b = None
comm.Reduce(b_temp, b, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Parallel dot: {b[0]} ... {b[-1]} (first and last element for verification)")
    print(f"Verification: {np.allclose(b, b_seq)}")
    print(f"Parallel time: {time.time() - start_time} seconds")
    np.savetxt('Results_parallel.dat', b, fmt='%.6f')
