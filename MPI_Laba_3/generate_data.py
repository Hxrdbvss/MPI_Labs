import numpy as np

N = 1000
M = 500
A = np.random.randn(N, M)  # Случайная матрица
b = np.random.randn(N)    # Случайный вектор b

np.savetxt('AData.dat', A.flatten())
np.savetxt('bData.dat', b)
with open('in.dat', 'w') as f:
    f.write(f"{N} {M}\n")

print("Данные сгенерированы")
