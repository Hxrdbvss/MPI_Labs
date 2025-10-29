from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.full((3, 3), rank, dtype=np.int32)
buffer = np.zeros((3, 3), dtype=np.int32)

times = []

for i in range(10):
    start = time.time()
    comm.Sendrecv_replace([data, MPI.INT], dest=(rank + 1) % size, sendtag=0,
                          source=(rank - 1) % size, recvtag=0, status=None)
    times.append(time.time() - start)

print(f"[Sendrecv] Rank {rank}: среднее время: {np.mean(times):.6f} сек")
