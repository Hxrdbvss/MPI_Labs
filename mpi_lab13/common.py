# common.py
import numpy as np
from mpi4py import MPI

def create_distributed_problem(N=8000):
    """
    Создаёт распределённую по строкам матрицу (1D-разбиение)
    Каждый процесс получает свои строки A_local и b_local
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Размер локальной части
    n_local = N // size
    if rank == size - 1:
        n_local += N % size

    # Локальная часть матрицы (плотная, но маленькая — 8000×8000 всего 512 МБ)
    A_local = np.zeros((n_local, N), dtype=np.float64)
    
    # Заполняем: диагональ + немного соседей (имитация сетки/лапласиана)
    for i in range(n_local):
        global_row = rank * (N // size) + i
        A_local[i, global_row] = 2.0 * N                     # главная диагональ
        if global_row > 0:
            A_local[i, global_row-1] = -1.0
        if global_row < N-1:
            A_local[i, global_row+1] = -1.0
        # добавим немного случайных элементов
        for _ in range(5):
            j = np.random.randint(0, N)
            if j != global_row:
                A_local[i, j] = np.random.rand() * 0.01

    # Правая часть b = A @ ones
    x_exact = np.ones(N, dtype=np.float64)
    b_local = A_local @ x_exact

    if rank == 0:
        print(f"Готово! Распределённая матрица {N}×{N}, локально {n_local} строк")

    return A_local, b_local, n_local
