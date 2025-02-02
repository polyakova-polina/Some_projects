import numpy as np
import matplotlib.pyplot as plt

# Данные
current = np.array([0, 0.18, 0.3, 1.16, 0.3, 0.38, 0.48, 0.58, 0.68, 0.81, 1.05, 1.29])
temperature = np.array([25.4, 24.9, 24.4, 29.5, 24.4, 24.1, 24.1, 24.1, 24.3, 24.8, 26.9, 32.5])

# Погрешности для каждой точки (примерные значения)
temperature_errors = np.array([0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.3, 0.3, 0.3, 0.2, 0.5, 0.5])

# Аппроксимация квадратичной зависимостью
coefficients = np.polyfit(current, temperature, 2)
quadratic = np.poly1d(coefficients)

# Создание массива для построения линии
x_fit = np.linspace(min(current), max(current), 100)
y_fit = quadratic(x_fit)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(current, temperature, color='blue', label='Данные')
plt.plot(x_fit, y_fit, color='red', label='Аппроксимация (квадратичная)', linewidth=2)

# Добавление крестов погрешностей
plt.errorbar(current, temperature, yerr=temperature_errors, fmt='o', color='blue', capsize=5)

# Настройка графика
plt.title('Зависимость температуры от тока')
plt.xlabel('Ток (А)')
plt.ylabel('Температура (К)')
plt.grid(True)
plt.legend()

# Вывод уравнения параболы на графике
equation_text = f'y = {coefficients[0]:.2f}x² - {-coefficients[1]:.2f}x + {coefficients[2]:.2f}'
plt.text(0.05, max(temperature) - 1, equation_text, fontsize=12)

# Показать график
plt.show()