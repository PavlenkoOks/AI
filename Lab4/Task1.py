import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.array([-1, -1, 0, 1, 2, 3])
Y = np.array([-1, 0, 1, 1, 3, 5])

def linear_func(x, a, b):
    return a * x + b

params, _ = curve_fit(linear_func, X, Y)
a, b = params

X_smooth = np.linspace(X.min() - 5, X.max() + 5, 500)
Y_smooth = linear_func(X_smooth, a, b)

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', s=50, edgecolor='black', label='Експериментальні дані')
plt.plot(X_smooth, Y_smooth, color='red', linewidth=2, label=f'Апроксимація: Y = {a:.2f}X + {b:.2f}')

plt.xlabel('X (незалежна змінна)', fontsize=12)
plt.ylabel('Y (залежна змінна)', fontsize=12)
plt.title('Експериментальні точки та апроксимуюча пряма', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
