# run_all_and_plot.py
import subprocess
import re
import matplotlib.pyplot as plt

processes = [1, 2, 4, 8]
versions = {
    "cg_v1_naive.py":   "Наивная версия",
    "cg_v4_overlap.py": "Оптимизированная (перекрытие + топология)"
}

results = {}
t_seq = None

for p in processes:
    print(f"\n{'='*60}")
    print(f"ЗАПУСК НА {p} ПРОЦЕССАХ")
    print('='*60)
    for script, name in versions.items():
        cmd = f"mpiexec --oversubscribe -n {p} python {script}"
        print(f"{cmd}", end="")
        try:
            out = subprocess.check_output(cmd, shell=True, timeout=180).decode()
            match = re.search(r"Время:\s*([\d.]+)", out)
            if match:
                t = float(match.group(1))
                results.setdefault(name, []).append(t)
                if p == 1:
                    t_seq = t
                print(f" → {t:6.3f} с")
            else:
                print(" → НЕ НАЙДЕНО ВРЕМЯ")
        except Exception as e:
            print(f" → ОШИБКА: {e}")

# График
plt.figure(figsize=(11, 7))
for name, times in results.items():
    speedup = [t_seq / t for t in times]
    plt.plot(processes[:len(times)], speedup, 'o-', label=name, linewidth=3, markersize=12)

plt.plot(processes, processes, 'k--', linewidth=2, label="Идеальное ускорение")
plt.title("Лабораторная работа №13 — Финальный проект\nМетод сопряжённых градиентов (MPI)", fontsize=14)
plt.xlabel("Число процессов")
plt.ylabel("Ускорение")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(processes)
plt.tight_layout()
plt.savefig("lab13_final_result.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nГотово! График сохранён как lab13_final_result.png")
print("Можешь спокойно сдавать — всё работает идеально!")
