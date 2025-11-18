import matplotlib.pyplot as plt

processes = [4, 9, 16]
times_cart = [0.1732, 0.1775, 0.4801]  # ЛР6
times_base_full = [0.0297, 0.0412, 0.1225]  # ЛР3 полная
times_base_simple = [0.0092, 0.0152, 0.0520]  # ЛР3 упрощённая

# Ускорение (относительно ЛР3 полной для 1 процесса: 0.0290)
speedup_base_full = [0.0290 / t for t in times_base_full]
speedup_base_simple = [0.0290 / t for t in times_base_simple]
speedup_cart = [0.0290 / t for t in times_cart]

# Эффективность
efficiency_base_full = [s / p for s, p in zip(speedup_base_full, processes)]
efficiency_base_simple = [s / p for s, p in zip(speedup_base_simple, processes)]
efficiency_cart = [s / p for s, p in zip(speedup_cart, processes)]

plt.figure(figsize=(10, 5))

# График ускорения
plt.subplot(1, 2, 1)
plt.plot(processes, speedup_base_full, label="Базовый CG полный (ЛР3)", marker='o')
plt.plot(processes, speedup_base_simple, label="Базовый CG упрощённый (ЛР3)", marker='o')
plt.plot(processes, speedup_cart, label="Топология (ЛР6)", marker='o')
plt.xlabel("Число процессов")
plt.ylabel("Ускорение")
plt.legend()

# График эффективности
plt.subplot(1, 2, 2)
plt.plot(processes, efficiency_base_full, label="Базовый CG полный (ЛР3)", marker='o')
plt.plot(processes, efficiency_base_simple, label="Базовый CG упрощённый (ЛР3)", marker='o')
plt.plot(processes, efficiency_cart, label="Топология (ЛР6)", marker='o')
plt.xlabel("Число процессов")
plt.ylabel("Эффективность")
plt.legend()

plt.tight_layout()
plt.savefig("performance_comparison.png")
