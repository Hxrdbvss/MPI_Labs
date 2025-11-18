# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import os

# Создаём папку images, если её нет
os.makedirs("images", exist_ok=True)

# Твои реальные замеры с большой матрицей (видно, что M и N теперь гораздо больше, чем 100×80)
procs = np.array([1, 4, 16, 25])
time  = np.array([0.011841, 0.005796, 0.009816, 0.034536])  # секунды

# Ускорение и эффективность
speedup     = time[0] / time
efficiency  = speedup / procs

# ===================================================================
# 1. График времени выполнения
# ===================================================================
plt.figure(figsize=(8, 5))
plt.plot(procs, time, marker='o', color='tab:blue', linewidth=2, markersize=8)
plt.title('Время выполнения от числа процессов', fontsize=14, pad=15)
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Время, сек', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(procs)
for i, t in enumerate(time):
    plt.text(procs[i], t + 0.0008, f'{t:.6f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('images/execution_time_lab5.png', dpi=300)
plt.close()

# ===================================================================
# 2. График ускорения
# ===================================================================
plt.figure(figsize=(8, 5))
plt.plot(procs, speedup, marker='s', color='tab:green', linewidth=2, markersize=8)
plt.plot(procs, procs, '--', color='gray', label='Линейное ускорение')  # идеальная линия
plt.title('Ускорение (Speedup)', fontsize=14, pad=15)
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Ускорение S(p)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(procs)
for i, s in enumerate(speedup):
    plt.text(procs[i], s + 0.15, f'{s:.2f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('images/speedup_lab5.png', dpi=300)
plt.close()

# ===================================================================
# 3. График эффективности
# ===================================================================
plt.figure(figsize=(8, 5))
plt.plot(procs, efficiency, marker='^', color='tab:red', linewidth=2, markersize=8)
plt.title('Эффективность параллелизма', fontsize=14, pad=15)
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Эффективность E(p) = S(p)/p', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(procs)
plt.ylim(0, 1.1)
for i, e in enumerate(efficiency):
    plt.text(procs[i], e + 0.03, f'{e:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('images/efficiency_lab5.png', dpi=300)
plt.close()

print("Графики успешно сохранены в папку images/:")
print("   • execution_time_lab5.png")
print("   • speedup_lab5.png")
print("   • efficiency_lab5.png")
