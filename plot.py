import matplotlib.pyplot as plt
import numpy as np

# Данные для графика 1: Время выполнения CG
procs = np.array([4, 9, 16])
time_sync = np.array([0.0727, 0.0550, 0.0523])
time_async = np.array([0.0681, 0.0482, 0.0415])

fig1, ax1 = plt.subplots()
ax1.plot(procs, time_sync, 'ro-', label='Синхронный')
ax1.plot(procs, time_async, 'gs-', label='Асинхронный')
ax1.set_xlabel('Процессов')
ax1.set_ylabel('Время, с')
ax1.set_title('Время выполнения CG')
ax1.legend()
ax1.grid(alpha=0.3)
plt.savefig('cg_time.png', dpi=300)
plt.close(fig1)

# Данные для графика 2: Ускорение CG
speedup = time_sync / time_async
fig2, ax2 = plt.subplots()
ax2.plot(procs, speedup, 'b^-', markersize=8)
ax2.set_xlabel('Процессов')
ax2.set_ylabel('Ускорение')
ax2.set_title('Ускорение CG')
ax2.grid(alpha=0.3)
plt.savefig('cg_speedup.png', dpi=300)
plt.close(fig2)

# Данные для графика 3: Сравнение кольцевого обмена
procs_ring = np.array([4, 9])
time_sendrecv = np.array([0.000812, 0.001245])
time_persistent = np.array([0.000623, 0.000789])

fig3, ax3 = plt.subplots()
ax3.plot(procs_ring, time_sendrecv, 'ro-', label='Sendrecv')
ax3.plot(procs_ring, time_persistent, 'gs-', label='Persistent')
ax3.set_xlabel('Процессов')
ax3.set_ylabel('Время, с')
ax3.set_title('Сравнение кольцевого обмена')
ax3.legend()
ax3.grid(alpha=0.3)
plt.savefig('ring_comparison.png', dpi=300)
plt.close(fig3)

print("Графики сохранены: cg_time.png, cg_speedup.png, ring_comparison.png")
