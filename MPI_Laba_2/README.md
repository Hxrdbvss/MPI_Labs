# Отчет о л/р №2: Коллективные операции MPI. Скалярное произведение и умножение транспонированной матрицы на вектор
Папка проекта (размещена в публичном репозитории)

# Файлы проекта: generate_data.py, dot.py (Часть 1), transpose_dot.py (Часть 2), readme.md (отчет).
Репозиторий: https://github.com/Hxrdbvss/MPI_Laba_-.

# Часть 1: Параллельное вычисление скалярного произведения
Описание задания
Напишите программу на Python с использованием mpi4py, которая вычисляет скалярное произведение двух одинаковых векторов (например,  a \cdot a ).

Инициализация и генерация данных: Процесс с рангом 0 создает вектор  a  длины  M  (например,  a = \text{numpy.arange}(1, M+1, \text{dtype=float64}) ).
Распределение данных: Используйте массивы rcounts и displs, чтобы корректно распределить части вектора  a  между всеми процессами с помощью MPI.Scatterv. Учтите случай, когда  M  не делится нацело на число процессов.
Локальные вычисления: Каждый процесс вычисляет скалярное произведение своей части вектора на себя:  \text{local\_dot} = \text{numpy.dot}(\text{a\_part}, \text{a\_part}) .
Глобальная редукция:
    Вариант А: Используйте MPI.Reduce с операцией MPI.SUM, чтобы собрать результат на процессе 0.
    Вариант Б: Используйте MPI.Allreduce с той же операцией, чтобы получить результат на всех процессах.
Верификация: Процесс 0 должен также вычислить скалярное произведение последовательно (с помощью numpy.dot(a, a)) и сравнить результат с параллельным вычислением для проверки правильности.

Листинг кода генерации файлов 
# generate_data.py

import numpy as np

M = 1000  
N = 1000

a = np.arange(1, M+1, dtype=np.float64)
np.savetxt('aData.dat', a, fmt='%.6f')

A = np.random.rand(M, N)
x = np.random.rand(M)

with open('in.dat', 'w') as f:
    f.write(f"{M} {N}\n")
np.savetxt('AData.dat', A, fmt='%.6f')
np.savetxt('xData.dat', x, fmt='%.6f')

Листинг кода 

# dot.py
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = time.time()
    a = np.loadtxt('aData.dat')
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
    print(f"Sequential dot: {dot_seq}")
    print(f"Parallel dot (Reduce): {global_dot_reduce}")
    print(f"Verification (Reduce): {global_dot_reduce == dot_seq}")
    print(f"Parallel time: {time.time() - start_time} seconds")
print(f"Rank {rank}: Parallel dot (Allreduce): {global_dot_allreduce}")

Генерация данных: M = 1000, вектор a сгенерирован в aData.dat.
Команда запуска: mpiexec -n 4 --oversubscribe mpi_env/bin/python dot.py
Вывод терминала (пример для 4 процессов):

Sequential dot: 3.3383358335e+08
Parallel dot (Reduce): 3.3383358335e+08
Verification (Reduce): True
Parallel time: 0.01567840576171875 seconds
Rank 0: Parallel dot (Allreduce): 3.3383358335e+08
Rank 1: Parallel dot (Allreduce): 3.3383358335e+08
Rank 2: Parallel dot (Allreduce): 3.3383358335e+08
Rank 3: Parallel dot (Allreduce): 3.3383358335e+08

Генерация данных: M = 1000, вектор a сгенерирован в aData.dat.
Команда запуска: mpiexec -n 2 --oversubscribe mpi_env/bin/python dot.py
Вывод терминала (пример для 2 процессов):

Sequential dot: 333833500.0
Parallel dot (Reduce): 333833500.0
Verification (Reduce): True
Parallel time: 0.0070362091064453125 seconds
Rank 0: Parallel dot (Allreduce): 333833500.0
Rank 1: Parallel dot (Allreduce): 333833500.0

Часть 2: Параллельное умножение транспонированной матрицы на вектор
Описание задания
Напишите программу, которая вычисляет  b = A^T \cdot x , где  A  – матрица размера  M \times N ,  x  – вектор длины  M .
Чтение входных данных: Процесс 0 считывает из файлов размеры  M  и  N , матрицу  A  и вектор  x .
Распределение данных:
Разбейте матрицу  A  на горизонтальные полосы (блоки строк). Каждый процесс получает блок  A_{\text{part}}  размером  \text{local\_M} \times N  с помощью MPI.Scatterv.
Согласованно разбейте вектор  x  на блоки длины  \text{local\_M}  и разошлите их с помощью MPI.Scatterv.
Локальные вычисления: Каждый процесс вычисляет произведение транспонированного своего блока матрицы на свой блок вектора:  b_{\text{temp}} = \text{numpy.dot}(A_{\text{part}}^T, x_{\text{part}}) . Результат  b_{\text{temp}}  – вектор длины  N .
Глобальная редукция: Итоговый вектор  b  является суммой векторов  b_{\text{temp}}  со всех процессов, используйте MPI.Reduce с операцией MPI.SUM, чтобы получить финальный вектор  b  на процессе 0.
Верификация: Процесс 0 должен вычислить результат последовательно ( b_{\text{seq}} = \text{numpy.dot}(A^T, x) ) и сравнить его с параллельным результатом.

Листинг кода 
# transpose_dot.py
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

Результаты:

Генерация данных: M = 1000, N = 1000, матрица A и вектор x сгенерированы в AData.dat, xData.dat, размеры в in.dat.
Команда запуска: mpiexec -n 4 --oversubscribe mpi_env/bin/python transpose_dot.py
Вывод терминала (пример для 4 процессов):
Sequential dot: 254.99386108819408 ... 251.73329314605985 (first and last element for verification)
Parallel dot: 254.993861088194 ... 251.73329314606002 (first and last element for verification)
Verification: True
Parallel time: 0.1409289836883545 seconds

Команда запуска: mpiexec -n 2 --oversubscribe mpi_env/bin/python transpose_dot.py
Вывод терминала (пример для 2 процессов):
Sequential dot: 254.99386108819408 ... 251.73329314606008 (first and last element for verification)
Parallel dot: 254.99386108819402 ... 251.73329314606002 (first and last element for verification)
Verification: True
Parallel time: 0.08176159858703613 seconds

# Выводы 
Обе части реализованы с обработкой произвольного  M . Алгоритмы используют коллективные операции MPI для распределения и редукции данных.
Производительность ограничена overhead на коммуникации для малого размера данных, что приводит к speedup < 1. Для больших  M  и  N  ожидаем значительное улучшение.
MPI.Reduce с MPI.SUM предпочтительнее MPI.Gatherv в Части 2, так как требуется суммирование, а не сбор всех векторов.
