from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Задание 2: массив из 10 элементов
data_send = np.full(10, rank, dtype=np.int32)
data_recv = np.zeros(10, dtype=np.int32)

requests = [MPI.REQUEST_NULL] * 2

# Определяем соседей в кольце
prev = (rank - 1) % size
next_proc = (rank + 1) % size

# Асинхронная отправка и приём
requests[0] = comm.Isend([data_send, MPI.INT], dest=prev, tag=0)
requests[1] = comm.Irecv([data_recv, MPI.INT], source=next_proc, tag=0)

# Задание 3: Добавляем вычисления между Isend/Irecv и Waitall
start_comp = time.time()
# Имитация вычислений: умножение матриц
A = np.random.rand(500, 500).astype(np.float64)
B = np.random.rand(500, 500).astype(np.float64)
C = A @ B  # Тяжёлая операция
comp_time = time.time() - start_comp

# Ожидание завершения коммуникаций
MPI.Request.Waitall(requests)

print(f"Rank {rank}: получил от {next_proc} данные: {data_recv}")
print(f"Rank {rank}: время вычислений: {comp_time:.4f} сек")
