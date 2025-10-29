from mpi4py import MPI
import numpy as np
import math

# -------------------------------------------------
def ring_sum(buf, comm_cart, left, right):
    """Кольцевое суммирование по горизонтали (dims[1] шагов)"""
    total = buf.copy()
    send = buf.copy()
    for _ in range(comm_cart.dims[1] - 1):
        recv = np.zeros_like(send)
        req_s = comm_cart.Isend([send, MPI.DOUBLE], dest=right, tag=99)
        req_r = comm_cart.Irecv([recv, MPI.DOUBLE], source=left, tag=99)

        # небольшая работа для перекрытия
        _ = np.sum(np.random.rand(200))

        MPI.Request.Waitall([req_s, req_r])
        total += recv
        send = recv
    return total

# -------------------------------------------------
def matrix_vector_multiply_async(A_part, x, comm_cart, left, right, n_local, n):
    Ax = np.dot(A_part, x)                     # локальное умножение
    return ring_sum(Ax, comm_cart, left, right)

def scalar_product_async(v, w, comm_cart, left, right, n_local):
    loc = np.dot(v, w)
    if not np.isfinite(loc): loc = 0.0
    send = np.array([loc], dtype=np.float64)
    return ring_sum(send, comm_cart, left, right)[0]

# -------------------------------------------------
def conjugate_gradient_async(A_part, b_loc, x, comm_cart, n_loc, n,
                            max_iter=200, tol=1e-5):
    left, right = comm_cart.Shift(1, 1)[0], comm_cart.Shift(1, 1)[1]
    rank = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(rank)
    start = coords[0] * n_loc

    r = b_loc - matrix_vector_multiply_async(A_part, x, comm_cart, left, right, n_loc, n)
    p = r.copy()
    p_full = np.zeros(n, dtype=np.float64)
    p_full[start:start+n_loc] = p
    comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)

    rsold = scalar_product_async(r, r, comm_cart, left, right, n_loc)

    for it in range(max_iter):
        Ap = matrix_vector_multiply_async(A_part, p_full, comm_cart, left, right, n_loc, n)
        pAp = scalar_product_async(p, Ap, comm_cart, left, right, n_loc)
        if abs(pAp) < 1e-12: break
        alpha = rsold / pAp

        x += alpha * p_full
        r -= alpha * Ap

        rsnew = scalar_product_async(r, r, comm_cart, left, right, n_loc)
        if np.sqrt(rsnew) < tol: break

        p = r + (rsnew/rsold)*p
        p_full[start:start+n_loc] = p
        comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)
        rsold = rsnew

    return x, it+1

# -------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root = int(math.sqrt(size))
if root*root != size:
    if rank == 0: print("Ошибка: число процессов должно быть квадратом!")
    exit()

dims = (root, root)
comm_cart = comm.Create_cart(dims, periods=(True, True), reorder=True)

n = 1200                         # чуть больше, чтобы было видно разницу
n_loc = n // root

np.random.seed(42)
A = np.random.rand(n, n)
A = (A + A.T)/2
A += 100*n*np.eye(n)

coords = comm_cart.Get_coords(comm_cart.Get_rank())
A_part = A[coords[0]*n_loc : (coords[0]+1)*n_loc, :]
b_loc = np.random.rand(n_loc)
x = np.zeros(n)

t0 = MPI.Wtime()
x, it = conjugate_gradient_async(A_part, b_loc, x, comm_cart, n_loc, n)
t1 = MPI.Wtime()

if comm_cart.Get_rank() == 0:
    print(f"ASYNC_CG_TIME: {t1-t0:.6f}")
