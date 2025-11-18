from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = MPI.Wtime()  
    a = np.loadtxt('aData_3.dat')
    M = a.size
    local_Ms = [M // size + (1 if i < M % size else 0) for i in range(size)]
    rcounts = local_Ms
    displs = [0] + np.cumsum(rcounts[:-1]).tolist()
else:
    a = None
    M = None
    local_Ms = None
    rcounts = None
    displs = None

M = comm.bcast(M, root=0)
local_M = comm.scatter(local_Ms, root=0)

a_part = np.empty(local_M, dtype=np.float64)
comm.Scatterv([a, rcounts, displs, MPI.DOUBLE], a_part, root=0)

local_dot = np.dot(a_part, a_part)

global_dot_reduce = comm.reduce(local_dot, op=MPI.SUM, root=0)
global_dot_allreduce = comm.allreduce(local_dot, op=MPI.SUM)

if rank == 0:
    dot_seq = np.dot(a, a)
    print(f"Parallel time: {MPI.Wtime() - start_time} seconds")  
