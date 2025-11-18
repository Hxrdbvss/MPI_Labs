import matplotlib.pyplot as plt

p = [1, 2, 4, 8, 16, 32, 64]
t_a = [1.003578, 1.464662, 1.655401, 2.104181, 6.368933, 6.127046, 15.162000]
t_b = [0.527212, 0.548649, 0.649504, 0.724006, 1.597187, 2.455667, 7.045032]
t_c = [0.137879, 0.124950, 0.274487, 0.419219, 0.966766, 1.856232, 3.897272]

speedup_a = [t_a[0] / t for t in t_a]
speedup_b = [t_b[0] / t for t in t_b]
speedup_c = [t_c[0] / t for t in t_c]
eff_a = [s / p[i] for i, s in enumerate(speedup_a)]
eff_b = [s / p[i] for i, s in enumerate(speedup_b)]
eff_c = [s / p[i] for i, s in enumerate(speedup_c)]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(p, t_a, label='A')
plt.plot(p, t_b, label='B')
plt.plot(p, t_c, label='C')
plt.xlabel('Processes')
plt.ylabel('Time (s)')
plt.title('Execution Time')
plt.legend()
plt.grid()
plt.savefig('images/execution_time.png')

plt.subplot(1, 3, 2)
plt.plot(p, speedup_a, label='A')
plt.plot(p, speedup_b, label='B')
plt.plot(p, speedup_c, label='C')
plt.xlabel('Processes')
plt.ylabel('Speedup')
plt.title('Speedup')
plt.legend()
plt.grid()
plt.savefig('images/speedup.png')
plt.subplot(1, 3, 3)
plt.plot(p, eff_a, label='A')
plt.plot(p, eff_b, label='B')
plt.plot(p, eff_c, label='C')
plt.xlabel('Processes')
plt.ylabel('Efficiency')
plt.title('Efficiency')
plt.legend()
plt.grid()
plt.savefig('images/efficiency.png')
plt.show()
