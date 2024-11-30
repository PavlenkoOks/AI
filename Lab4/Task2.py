import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

coefficients = np.polyfit(x, y, 4)
polynomial = np.poly1d(coefficients)

y_02 = polynomial(0.2)
y_05 = polynomial(0.5)

print("Коефіцієнти полінома:", coefficients)
print(f"Значення функції в точці x = 0.2: {y_02:.3f}")
print(f"Значення функції в точці x = 0.5: {y_05:.3f}")

x_range = np.linspace(min(x) - 0.05, max(x) + 0.05, 500)
y_range = polynomial(x_range)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', color='blue', markersize=8, label='Табличні точки', markeredgecolor='black')
plt.plot(x_range, y_range, '-', color='red', linewidth=2, label='Інтерполяційний поліном')

plt.xlabel('x (незалежна змінна)', fontsize=12)
plt.ylabel('y (залежна змінна)', fontsize=12)
plt.title('Інтерполяція поліномом 4-го степеня', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
