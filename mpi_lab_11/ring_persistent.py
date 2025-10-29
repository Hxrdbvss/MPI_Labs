from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Задание 2: 2D массив 3x3
data = np.full((3, 3), rank, dtype=np.int32)
data_recv = np.zeros((3, 3), dtype=np.int32)

# Отложенные запросы
send_req = comm.Send_init([data, MPI.INT], dest=(rank + 1) % size, tag=0)
recv_req = comm.Recv_init([data_recv, MPI.INT], source=(rank - 1) % size, tag=0)

times = []

for i in range(10):
    start = time.time()

    MPI.Prequest.Startall([send_req, recv_req])
    MPI.Request.Waitall([send_req, recv_req])

    # Обновляем данные: теперь отправляем полученное
    data[:] = data_recv

    times.append(time.time() - start)

    if i == 9:
        print(f"Rank {rank}: финальное значение:\n{data}")

print(f"Rank {rank}: среднее время итерации: {np.mean(times):.6f} сек")
