import matplotlib.pyplot as plt

ps = [1, 2, 4, 8]
times_full = [0.0290, 0.0256, 0.0297, 0.0368]  
times_simp = [0.0129, 0.0099, 0.0092, 0.0138]  

speedup_full = [times_full[0] / t if t > 0 else 1 for t in times_full]
eff_full = [s / p for s, p in zip(speedup_full, ps)]

plt.figure()
plt.plot(ps, speedup_full, marker='o')
plt.xlabel('Процессы')
plt.ylabel('Ускорение')
plt.title('Ускорение (full)')
plt.grid()
plt.savefig('speedup_full.png')

plt.figure()
plt.plot(ps, eff_full, marker='o')
plt.xlabel('Процессы')
plt.ylabel('Эффективность')
plt.title('Эффективность (full)')
plt.grid()
plt.savefig('efficiency_full.png')

speedup_simp = [times_simp[0] / t if t > 0 else 1 for t in times_simp]
eff_simp = [s / p for s, p in zip(speedup_simp, ps)]

plt.figure()
plt.plot(ps, speedup_simp, marker='o')
plt.xlabel('Процессы')
plt.ylabel('Ускорение')
plt.title('Ускорение (simp)')
plt.grid()
plt.savefig('speedup_simp.png')

plt.figure()
plt.plot(ps, eff_simp, marker='o')
plt.xlabel('Процессы')
plt.ylabel('Эффективность')
plt.title('Эффективность (simp)')
plt.grid()
plt.savefig('efficiency_simp.png')
