import numpy as np

M = 1000000  
N = 500

a = np.arange(1, M+1, dtype=np.float64)
np.savetxt('aData_3.dat', a, fmt='%.6f')

A = np.random.rand(M, N)
x = np.random.rand(M)

with open('in.dat', 'w') as f:
    f.write(f"{M} {N}\n")
np.savetxt('AData_3.dat', A, fmt='%.6f')
np.savetxt('xData_3.dat', x, fmt='%.6f')
