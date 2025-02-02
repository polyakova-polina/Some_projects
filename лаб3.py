import numpy as np
import matplotlib.pyplot as plt

# Ввод данных эксперимента (давление в Торр, сила тока в пА)
pressure1 = np.array(
    [0.224, 0.207, 0.414, 0.602, 0.834, 1.043, 1.249, 1.414, 1.645, 0.102, 0.05, 0.025, -0.025, -0.105])
pressure2 = np.array([0.234, 0.447, 0.649, 0.864, 1.064, 1.261, 1.456, 0.107, 0.026, -0.04, -0.261])
pressure3 = np.array([0.307, 0.536, 0.727, 0.953, 1.119, 1.378, 1.62, 1.837, 0.081, 0.026, -0.007, -0.007])
pressure4 = np.array([0.114, 0.307, 0.53, 0.725, 0.913, 1.13, 1.352, 1.543, 0.027, -0.014, -0.085, -0.211, -0.302])

pressure = np.array([pressure1, pressure2, pressure3, pressure4], dtype=object)

current1 = np.array([0.071, 0.071, 0.089, 0.105, 0.134, 0.158, 0.187, 0.207, 0.239, 0.071, 0.063, 0.063, 0.055, 0.032])
current2 = np.array(
    [0.083666, 0.1, 0.13038405, 0.15491933, 0.18973666, 0.21679483, 0.24899799, 0.05477226, 0.05477226, 0.03162278, 0])
current3 = np.array(
    [0.07071068, 0.09486833, 0.1183216, 0.15165751, 0.19235384, 0.23664319, 0.2792848, 0.32403703, 0.05477226,
     0.05477226, 0.04472136, 0.03162278])
current4 = np.array(
    [0.05477226, 0.05477226, 0.06324555, 0.07745967, 0.10488088, 0.14142136, 0.18708287, 0.23021729, 0.05477226,
     0.04472136, 0.04472136, 0.03162278, 0])

current = np.array([current1, current2, current3, current4], dtype=object)

a = np.array([0.1233, 0.1469, 0.1862, 0.1871])  # Коэффициент наклона первой прямой
b = np.array([0.033, 0.0329, -0.0199, -0.0639])
lamb = np.array([585.2, 621.7, 659.9, 692.9])

# Определение линейной функции
def line1(x, a1, b1):
    return a1 * x + b1


def find_intersection():
    a1 = 0.0934
    b1 = 0.013
    a2 = 0
    b2 = 934.1
    intersection = []
    intersection.append((b2 - b1) / (a1 - a2))
    intersection.append(a1 * intersection[0] + b1)
    return np.array(intersection)


# Построение графиков
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Создаем 2x2 подграфика
axs = axs.flatten()  # Упрощаем доступ к подграфикам

x_values = np.linspace(-0.182, 2, 100)

for i in range(4):
    # Построим точки экспериментальных данных
    axs[i].scatter(pressure[i], current[i], color='blue', s=15)

    # Параметры для осей с указанием единиц измерения
    axs[i].set_xlabel('V (В)', fontsize=12)
    axs[i].set_ylabel('I**0.5 ', fontsize=12)

    # Построение двух прямых
    axs[i].plot(x_values, line1(x_values, a[i], b[i]), color='red')

    intersection = find_intersection()

    # Добавление заголовка графика
    axs[i].set_title(f'lambda = {lamb[i]} нм',fontsize=14)

    # Добавление сетки
    axs[i].grid(True)

# Удаление лишних верхних и правых границ графиков
for ax in axs:
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Добавление стрелок к осям
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Добавление легенды
for ax in axs:
    ax.legend()

# Показ графика
plt.tight_layout()  # Подгонка подграфиков, чтобы они не пересекались
plt.show()