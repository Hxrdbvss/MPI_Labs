import numpy as np

M = 1000  # Можно изменить для тестов
N = 1000

a = np.arange(1, M+1, dtype=np.float64)
np.savetxt('aData.dat', a, fmt='%.6f')

A = np.random.rand(M, N)
x = np.random.rand(M)

with open('in.dat', 'w') as f:
    f.write(f"{M} {N}\n")
np.savetxt('AData.dat', A, fmt='%.6f')
np.savetxt('xData.dat', x, fmt='%.6f')
