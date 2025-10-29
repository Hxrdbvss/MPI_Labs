# analyze.py
import numpy as np
import matplotlib.pyplot as plt

# Последовательное время
serial = np.load("serial_result.npz")['time']
T_serial = serial

procs_1d = [1, 2, 4, 8, 16]
times_1d = []
for p in procs_1d:
    try:
        t = np.load(f"mpi1d_{p}.npz")['time']
        times_1d.append(t)
    except:
        times_1d.append(np.nan)

procs_2d = [1, 4, 9, 16]
times_2d = []
for p in procs_2d:
    try:
        t = np.load(f"mpi2d_{p}.npz")['time']
        times_2d.append(t)
    except:
        times_2d.append(np.nan)

# Ускорение и эффективность
S1 = [T_serial / t for t in times_1d]
E1 = [s / p for s, p in zip(S1, procs_1d)]

S2 = [T_serial / t for t in times_2d]
E2 = [s / p for s, p in zip(S2, procs_2d)]

# Графики
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(procs_1d, times_1d, 'o-', label='1D')
plt.plot(procs_2d, times_2d, 's-', label='2D')
plt.axhline(T_serial, color='r', linestyle='--', label='Serial')
plt.xlabel('Число процессов')
plt.ylabel('Время (с)')
plt.legend()
plt.title('Время выполнения')

plt.subplot(2, 2, 2)
plt.plot(procs_1d, S1, 'o-', label='1D')
plt.plot(procs_2d, S2, 's-', label='2D')
plt.plot(procs_1d, procs_1d, 'k--')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.legend()
plt.title('Ускорение')

plt.subplot(2, 2, 3)
plt.plot(procs_1d, E1, 'o-', label='1D')
plt.plot(procs_2d, E2, 's-', label='2D')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность')
plt.legend()
plt.title('Эффективность')

plt.tight_layout()
plt.savefig("performance.png")
plt.show()
