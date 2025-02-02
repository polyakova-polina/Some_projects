import numpy as np
import matplotlib.pyplot as plt

# Ввод данных эксперимента (давление в Торр, сила тока в пА)
Y = np.array([0.433, 0.975, 1.221, 1.295, 2.259, 2.213, 2.089, 1.832, 1.646, 1.435, 1.354, 1.229, 0.629, 0.345, 0.02 ])  # Пример давления

X = np.array([6.91, 7.02, 7.11, 7.19, 7.55, 7.50, 7.44, 7.38, 7.31, 7.24, 7.17, 7.09, 7, 6.91, 6.8])
 # Пример силы тока

dI = np.array([0, 0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.008, 0.009, 0.008
])

dN = np.array([0.02, 0.03, 0.03, 0.004, 0.004, 0.005, 0.006, 0.009, 0.124, 0.16
])  # Задайте ваши погрешности по x (dI)

# Определение двух уравнений прямых
# Пример: y = a*x + b для обеих прямых
def line1(x):
    a1 = 0  # Коэффициент наклона первой прямой
    b1 = 0  # Смещение первой прямой
    return a1 * x + b1

def line2(x):
    a2 = 0  # Коэффициент наклона второй прямой
    b2 = 1   # Смещение второй прямой
    return a2 * x + b2

def find_intersection(m, c):
    a1 = m
    b1 = c
    a2 = 0  # Коэффициент наклона второй прямой
    b2 = 1
    intersection = []
    intersection.append((b2-b1)/(a1-a2))
    intersection.append(a1 * intersection[0] + b1)
    return np.array(intersection)

def find_intersection2(m, c):
    a1 = m
    b1 = c
    a2 = 0  # Коэффициент наклона второй прямой
    b2 = 0
    intersection = []
    intersection.append((b2-b1)/(a1-a2))
    intersection.append(a1 * intersection[0] + b1)
    return np.array(intersection)

# Добавление столбца единиц для расчетов
A = np.vstack([X, np.ones(len(X))]).T

# Выполнение наименьших квадратов
def line3(x, A = A, Y = Y):
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(m, c)
    return m * x + c

# Построение графика
fig, ax = plt.subplots()
# Построим точки экспериментальных данных
ax.scatter(X, Y, color='blue', s=15, label='Экспериментальные данные')

# Параметры для осей с указанием единиц измерения
ax.set_xlabel('ln(T)', fontsize=12)
ax.set_ylabel('ln(W) ', fontsize=12)
x_values = np.linspace(6.8, 7.6, 100)

# Построение двух прямых
#ax.plot(x_values, line1(x_values), color='red', label='Прямая 1')
#ax.plot(x_values, line2(x_values), color='red', label='Прямая 1')
#ax.plot(x_values, line3(x_values), color='red', label='Прямая 1')
ax.plot(x_values, line3(x_values), color='red', label='Прямая 2')

m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
#intersection = find_intersection(m, c)
#intersection2 = find_intersection2(m, c)

# Отметка точки пересечения прямых
#ax.scatter(intersection[0], intersection[1], color='purple', zorder=5, label=f'Точка пересечения\n({intersection[0]:.2f}, {intersection[1]:.2f})')
#ax.scatter(intersection2[0], intersection2[1], color='purple', zorder=5, label=f'Точка пересечения\n({intersection[0]:.2f}, {intersection[1]:.2f})')
#ax.set_ylim(0, 0.226)  # Границы по оси X
#ax.set_xlim(0, 0.048)
# Добавление стрелок к осям
#ax.spines['left'].set_position(('data', 50))  # Смещение оси Y
#ax.spines['bottom'].set_position(('data', 15))  # Смещение оси X

# Удаление верхней и правой границ графика
#ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Добавление стрелок к осям
#ax.plot(15, 720, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # Стрелка по оси X
#ax.plot(50, 322, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # Стрелка по оси Y
#ax.errorbar(X, Y, xerr=dI, yerr=dN, fmt='|', color='blue', label='Погрешности', capsize=5)

# Добавление легенды
#ax.legend()

# Показ графика
plt.grid(True)
plt.show()