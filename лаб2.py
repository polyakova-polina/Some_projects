import numpy as np
import matplotlib.pyplot as plt

# Ввод данных эксперимента (давление в Торр, сила тока в пА)

pressure = np.array([
0,
4.38,
8.7,
13.08,
17.74,
22.62,
27.32,
31.7])  # Пример давления

current = np.array([
0,
0.61333503,
1.14544206,
1.63334575,
2.13712145,
2.66419449,
3.15690023,
3.77380428



])  # Пример силы тока
'''
# Определение двух уравнений прямых
# Пример: y = a*x + b для обеих прямых

pressure = np.array([
0,
19.98,
39.96,
59.96,
79.94,
100.08,
119.52,
139.46,
159.26
])  # Пример давления

current = np.array([
0,
0.49376215,
0.89666687,
1.33328072,
1.69762504,
2.08683963,
2.48708736,
2.85070524,
3.20001016
])
'''
def line1(x):
    a1 = 0.1153  # Коэффициент наклона первой прямой
    b1 = 0.081   # Смещение первой прямой
    return a1 * x + b1

def line2(x):
    a2 = 0  # Коэффициент наклона второй прямой
    b2 = 934.1    # Смещение второй прямой
    return a2 * x + b2

def find_intersection():
    a1 = 0.0934
    b1 = 0.013
    a2 = 0  # Коэффициент наклона второй прямой
    b2 = 0
    intersection = []
    intersection.append((b2-b1)/(a1-a2))
    intersection.append(a1 * intersection[0] + b1)
    return np.array(intersection)

# Построение графика
fig, ax = plt.subplots()
# Построим точки экспериментальных данных
ax.scatter(pressure, current, color='blue', s=10, label='Экспериментальные данные')

# Параметры для осей с указанием единиц измерения
ax.set_xlabel('l, мм', fontsize=12)
ax.set_ylabel('ln(N0/N) ', fontsize=12)
x_values = np.linspace(0, 32, 100)

# Построение двух прямых
ax.plot(x_values, line1(x_values), color='red', label='Прямая 1')
#ax.plot(x_values, line2(x_values), color='red', label='Прямая 2')

intersection = find_intersection()
# Отметка точки пересечения прямых
#ax.scatter(intersection[0], intersection[1], color='purple', zorder=5, label=f'Точка пересечения\n({intersection[0]:.2f}, {intersection[1]:.2f})')

# Добавление стрелок к осям
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# Удаление верхней и правой границ графика
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Добавление стрелок к осям
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Добавление легенды
#ax.legend()

# Показ графика
plt.grid(True)
plt.show()