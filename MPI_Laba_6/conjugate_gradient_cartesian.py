from mpi4py import MPI
import numpy as np
import math

def matrix_vector_multiply(A_part, x, comm_cart, neighbour_left, neighbour_right, n_local, n):
    # Локальное умножение матрицы на вектор
    Ax_local = np.dot(A_part, x)  # (n_local, n) @ (n,) -> (n_local,)
    
    # Горизонтальное суммирование результатов
    send_buf = Ax_local.copy()
    total_Ax = Ax_local.copy()
    for _ in range(comm_cart.dims[1] - 1):
        comm_cart.Sendrecv_replace(send_buf, dest=neighbour_right, source=neighbour_left)
        total_Ax += send_buf
    
    return total_Ax

def scalar_product(v_local, w_local, comm_cart, neighbour_left, neighbour_right, n_local):
    # Локальное скалярное произведение
    local_dot = np.dot(v_local, w_local)
    
    # Проверка на NaN или inf
    if not np.isfinite(local_dot):
        local_dot = 0.0
    
    # Горизонтальное накопление
    send_buf = np.array([local_dot], dtype=np.float64)
    total_dot = send_buf.copy()
    for _ in range(comm_cart.dims[1] - 1):
        comm_cart.Sendrecv_replace(send_buf, dest=neighbour_right, source=neighbour_left)
        total_dot += send_buf
        if not np.isfinite(total_dot):
            total_dot = np.array([0.0], dtype=np.float64)
    
    return total_dot[0]

def conjugate_gradient_method(A_part, b_local, x, comm_cart, n_local, n, max_iter=1000, tol=1e-4):
    # Определение соседей
    neighbour_up, neighbour_down = comm_cart.Shift(0, 1)
    neighbour_left, neighbour_right = comm_cart.Shift(1, 1)
    rank = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(rank)
    
    # Инициализация
    r_local = b_local - matrix_vector_multiply(A_part, x, comm_cart, neighbour_left, neighbour_right, n_local, n)
    p = r_local.copy()  # p имеет размер (n_local,)
    
    # Синхронизация p между процессами
    p_full = np.zeros(n, dtype=np.float64)
    start_idx = coords[0] * n_local
    p_full[start_idx:start_idx + n_local] = p
    comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)
    
    rsold = scalar_product(r_local, r_local, comm_cart, neighbour_left, neighbour_right, n_local)
    if rank == 0:
        print(f"Итерация 0: rsold = {rsold}")
    
    for iter in range(max_iter):
        Ap_local = matrix_vector_multiply(A_part, p_full, comm_cart, neighbour_left, neighbour_right, n_local, n)
        pAp = scalar_product(p, Ap_local, comm_cart, neighbour_left, neighbour_right, n_local)
        
        # Проверка деления на ноль
        if abs(pAp) < 1e-10:
            if rank == 0:
                print(f"Итерация {iter}: pAp = {pAp}, деление на ноль, останавливаем")
            break
        
        alpha = rsold / pAp
        if not np.isfinite(alpha):
            if rank == 0:
                print(f"Итерация {iter}: alpha = {alpha}, останавливаем")
            break
        
        # Обновление x и r_local
        x += alpha * p_full
        r_local -= alpha * Ap_local
        
        rsnew = scalar_product(r_local, r_local, comm_cart, neighbour_left, neighbour_right, n_local)
        if rank == 0:
            print(f"Итерация {iter + 1}: rsnew = {rsnew}, alpha = {alpha}")
        
        # Проверка сходимости
        if not np.isfinite(rsnew) or np.sqrt(rsnew) < tol:
            break
            
        # Обновление p
        p = r_local + (rsnew / rsold) * p
        p_full[start_idx:start_idx + n_local] = p
        comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)
        rsold = rsnew
    
    return x, iter + 1

# Основная программа
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Проверка на квадратное число процессов
root = int(math.sqrt(size))
if root * root != size:
    if rank == 0:
        print("Ошибка: Число процессов должно быть квадратом!")
    MPI.Finalize()
    exit()

# Создание декартовой топологии
dims = (root, root)
periods = (True, True)
comm_cart = comm.Create_cart(dims=dims, periods=periods, reorder=True)
coords = comm_cart.Get_coords(comm_cart.Get_rank())

# Параметры задачи
n = 100  # Размер матрицы
n_local = n // root  # Локальный размер на процесс

# Генерация симметричной положительно определённой матрицы
np.random.seed(42)  # Для воспроизводимости
A_full = np.random.rand(n, n)
A_full = (A_full + A_full.T) / 2  # Симметризация
A_full += 100 * n * np.eye(n)  # Диагональное доминирование
start_idx = coords[0] * n_local
A_part = A_full[start_idx:start_idx + n_local, :]  # Локальная часть матрицы
b_local = np.random.rand(n_local)  # Локальная часть вектора правой части
x = np.zeros(n)  # Полный вектор x

# Замер времени
start_time = MPI.Wtime()

# Выполнение метода сопряжённых градиентов
x, iterations = conjugate_gradient_method(A_part, b_local, x, comm_cart, n_local, n)

# Замер времени
end_time = MPI.Wtime()

# Вывод результатов
if comm_cart.Get_rank() == 0:
    print(f"Процесс {comm_cart.Get_rank()} (coords {coords}): Итераций = {iterations}, Время = {end_time - start_time:.4f} сек")
