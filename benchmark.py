# benchmark.py
# Полный бенчмарк: 1, 2, 4, 8, 16 процессов
# Автоматически добавляет --oversubscribe при >4

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# === Параметры (должны совпадать с serial и parallel) ===
N = 10000
M = 2500

procs_list = [1, 2, 4, 8, 16]
times = []
speedups = []
efficiencies = []

print(f"Starting benchmark: N={N}, M={M}")
print("="*60)

for p in procs_list:
    print(f"\nRunning with {p} process{'es' if p > 1 else ''}...", end="")
    
    # Формируем команду
    if p == 1:
        cmd = ["python", "serial_rosenbrock.py"]
    else:
        # Добавляем --oversubscribe для >4
        oversub = ["--oversubscribe"] if p > 4 else []
        cmd = ["mpirun"] + oversub + ["-np", str(p), "python", "parallel_rosenbrock.py"]
    
    # Запуск
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Проверка ошибок
    if result.returncode != 0:
        print(f"\nMPI Error (code {result.returncode}):\n{result.stderr}")
        t = 0.0
    else:
        # Ищем время
        if p == 1:
            match = re.search(r"Serial computation completed in ([\d.]+) seconds", result.stdout)
            t = float(match.group(1)) if match else 0.0
        else:
            match = re.search(rf"Parallel time \({p} procs\): ([\d.]+) sec", result.stdout)
            t = float(match.group(1)) if match else 0.0
        print(f" Time: {t:.3f}s")
    
    times.append(t)

# === Анализ ===
times = np.array(times)
speedup = times[0] / np.where(times > 0, times, 1e-10)
efficiency = speedup / np.array(procs_list)

# === Вывод таблицы ===
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)
print(f"{'Procs':>6} | {'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10}")
print("-"*70)
for p, t, s, e in zip(procs_list, times, speedup, efficiency):
    print(f"{p:>6} | {t:>10.3f} | {s:>8.2f} | {e:>10.2f}")
print("="*70)

# === Графики ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(procs_list, times, 'o-', color='red', label='Time')
plt.xlabel('Number of Processes')
plt.ylabel('Time (s)')
plt.title('Execution Time')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(procs_list, speedup, 'o-', color='blue', label='Speedup')
plt.plot(procs_list, procs_list, '--', color='gray', label='Linear')
plt.plot(procs_list, efficiency, 's-', color='green', label='Efficiency')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup / Efficiency')
plt.title('Speedup & Efficiency')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("performance_analysis.png", dpi=200)
plt.close()

print(f"Plot saved: performance_analysis.png")
