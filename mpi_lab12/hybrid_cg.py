# hybrid_cg.py
import numpy as np
from mpi4py import MPI
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === НАСТРОЙКИ ===
N_local = 12000          # Локальный размер матрицы (на процесс)
REPEATS = 20             # Количество повторений для точности
# =================

N = N_local * size
omp_threads = int(os.environ.get("OMP_NUM_THREADS", "N/A"))

# Вывод только от root
if rank == 0:
    print(f"{'='*50}")
    print(f"ГИБРИДНЫЙ ТЕСТ: MPI + OpenMP")
    print(f"Узлы (MPI процессов): {size}")
    print(f"OpenMP потоков на процесс: {omp_threads}")
    print(f"Глобальный размер матрицы: {N} x {N}")
    print(f"Повторений matvec: {REPEATS}")
    print(f"{'='*50}")

# Локальные данные
np.random.seed(rank + 100)
A_local = np.random.rand(N_local, N_local).astype(np.float64)
x_local = np.random.rand(N_local).astype(np.float64)

# Рассылка вектора x
if rank == 0:
    x_global = np.random.rand(N).astype(np.float64)
else:
    x_global = np.empty(N, dtype=np.float64)
comm.Bcast(x_global, root=0)
x_local = x_global[rank * N_local : (rank + 1) * N_local]

# Прогрев (чтобы BLAS "разогрелся")
for _ in range(3):
    _ = A_local @ x_local
comm.Barrier()

# === ОСНОВНОЙ ЦИКЛ ===
t_start = time.time()
for _ in range(REPEATS):
    y_local = A_local @ x_local  # ← OpenMP внутри BLAS!
comm.Barrier()
t_end = time.time()
# =====================

elapsed = (t_end - t_start) / REPEATS
flops = 2.0 * N * N
gflops = flops / elapsed / 1e9

# Сбор данных
times = comm.gather(elapsed, root=0)
gflops_list = comm.gather(gflops, root=0)

if rank == 0:
    avg_time = sum(times) / len(times)
    total_flops = flops * REPEATS
    total_gflops = total_flops / (t_end - t_start) / 1e9
    print(f"Среднее время matvec: {avg_time:.4f} сек")
    print(f"Производительность: {total_gflops:.2f} GFLOPS")
    print(f"Проверка: norm(y) ≈ {np.linalg.norm(A_local @ x_local):.2f}")
    print(f"{'='*50}\n")
