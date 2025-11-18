# generate_data.py
import numpy as np

def generate_data(M=100, N=100, seed=42):
    """
    Генерирует тестовые данные для лабораторной:
    - in.dat: M N
    - AData.dat: матрица A (M x N)
    - xData.dat: вектор x (N,)
    - bData.dat: вектор b = A @ x (M,)
    """
    np.random.seed(seed)
    
    # --- 1. in.dat ---
    with open('in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    print(f"Создан in.dat: M={M}, N={N}")
    
    # --- 2. AData.dat ---
    A = np.random.rand(M, N) * 10  # случайная матрица
    np.savetxt('AData.dat', A.flatten(), fmt='%.15f')
    print(f"Создан AData.dat: {M} x {N} = {M*N} элементов")
    
    # --- 3. xData.dat ---
    x = np.random.rand(N) * 5
    np.savetxt('xData.dat', x, fmt='%.15f')
    print(f"Создан xData.dat: {N} элементов")
    
    # --- 4. bData.dat ---
    b = A @ x
    np.savetxt('bData.dat', b, fmt='%.15f')
    print(f"Создан bData.dat: b = A @ x, {M} элементов")
    
    # --- Проверка ---
    print(f"||b|| = {np.linalg.norm(b):.6f}")
    print(f"||x|| = {np.linalg.norm(x):.6f}")

if __name__ == "__main__":
    # Можно менять размеры
    generate_data(M=100, N=80, seed=123)
