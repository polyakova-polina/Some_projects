import numpy as np
import matplotlib.pyplot as plt

# Ввод данных эксперимента (давление в Торр, сила тока в пА)
pressure = np.array([10, 20, 25, 16, 17, 18, 19, 18.5, 17.5, 18.25, 14, 17.75, 15, 17.5, 18.25, 17.75, 18.5, 19, 18])
current = np.array([7.13559322, 0.2625, 0.123076923, 6.966666667, 6.741935484, 3.935483871, 0.885245902, 1.557377049,
                    5.714285714, 3.382352941, 7.288135593, 5.06557377, 7.315789474, 5.714285714, 3.382352941, 5.06557377,
                    1.557377049, 0.885245902, 3.935483871])

# Определение первой прямой
def line1(x):
    a1 = -3.4542  # Коэффициент наклона первой прямой
    b1 = 66.175   # Смещение первой прямой
    return a1 * x + b1

# Поиск точки пересечения
def find_intersection():
    a1 = -3.4542
    b1 = 66.175
    a2 = 0
    b2 = 0
    intersection = []
    intersection.append((b2 - b1) / (a1 - a2))  # X координата пересечения
    intersection.append(a1 * intersection[0] + b1)  # Y координата пересечения
    return np.array(intersection)

# Построение графика
fig, ax = plt.subplots()

# Построим точки экспериментальных данных
ax.scatter(pressure, current, color='red', s=10, label='Экспериментальные данные')

# Параметры для осей с указанием единиц измерения
ax.set_xlabel('x, мм', fontsize=12)
ax.set_ylabel('Кол-во счетов, 1/с', fontsize=12)

x_values = np.linspace(17, 19.5, 100)
y_values = np.linspace(0, 8, 100)

# Построение первой прямой
ax.plot(x_values, line1(x_values), color='green', label='Прямая 1')

# Добавляем уравнение первой прямой на график
equation_text = r'$y = -3.45x + 66.18$'
ax.text(18.5, 6.5, equation_text, fontsize=10, color='green')

# Находим и отмечаем точку пересечения
intersection = find_intersection()
ax.scatter(intersection[0], intersection[1], color='purple', zorder=5, label=f'Точка пересечения\n({intersection[0]:.2f}, {intersection[1]:.2f})')

# Обозначаем точку пересечения на графике с помощью стрелки
ax.annotate(f'({intersection[0]:.2f}, {intersection[1]:.2f})',
            xy=(intersection[0], intersection[1]),
            xytext=(intersection[0] + 0.1, intersection[1] + 0.5))


# Добавление стрелок к осям
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# Удаление верхней и правой границ графика
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Добавление стрелок к осям
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Включение сетки и отображение графика
plt.grid(True)
plt.show()
