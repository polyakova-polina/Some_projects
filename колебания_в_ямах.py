import matplotlib.pyplot as plt
import numpy as np

# Создать графики
def sred(x,y, par):
    y = y[:len(y) - len(y) % par]
    return x[:len(x) - par:par], (np.reshape(y, (len(y) // par, par))).sum(axis=1) / par

# Считаем интеграл
def integral(X,Y):
    iy =[]
    X = np.array(X)
    #print(X[2])
    dx = (X[len(X) - 1] - X[0]) / len(X)
    int_sum = 0
    for y in Y:
        int_sum += y * dx
        iy.append(int_sum)
    return int_sum

E = 3

# Строим периуды
def qraf():
    ans1 = []
    ans2 = []
    ans3 = []
    ans4 = []
    ans = []
    E1 = []
    E2 = []
    E3 = []
    E0 = []
    E_1 = np.linspace(-125 / 16, 4, 100)
    E_2 = np.linspace(4, 10, 100)

    #E_ = np.concatenate((E_1, E_2))
    E_ = np.linspace(-125 / 16, 20 , 500)

    for E in E_:
        ar = []
        delt = 0.001
        prev = -100
        for x_ in np.linspace(-5, 5, 100000):
            if abs(3 * x_ ** 4 - 2 * x_ ** 3 - 9 * x_ ** 2 + 4 - E) < delt and abs(prev - x_) > 0.1:
                ar.append(x_)
                prev = x_
        if len(ar) == 4:

            E1.append(E)
            E0.append(E)
            E0.append(E)
            d1, u1 = ar[0], ar[1]
            d2, u2 = ar[2], ar[3]

            x = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
            y = 2 / (2 * (E - 3 * x ** 4 + 2 * x ** 3 + 9 * x ** 2 - 4)) ** 0.5

            x2 = np.linspace(d2 + 0.001, u2 - 0.001, 10000)
            y2 = 2 / (2 * (E - 3 * x2 ** 4 + 2 * x2 ** 3 + 9 * x2 ** 2 - 4)) ** 0.5
            # print(d,u, len(ar))

            int_s1 = integral(x, y)
            int_s2 = integral(x2, y2)
            ans1.append(int_s1)
            ans2.append(int_s2)
            ans.append(int_s1)
            ans.append(int_s2)

        elif len(ar) == 2:
            E2.append(E)
            E0.append(E)
            d1, u1 = ar[0], ar[1]

            x3 = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
            y3 = 2 / (2 * (E - 3 * x3 ** 4 + 2 * x3 ** 3 + 9 * x3 ** 2 - 4)) ** 0.5

            int_s3 = integral(x3, y3)
            ans3.append(int_s3)
            ans.append(int_s3)

        elif len(ar) == 3:
            E3.append(E)
            E0.append(E)
            d1, u1 = ar[0], ar[2]

            x4 = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
            y4 = 2 / (2 * (E - 3 * x4 ** 4 + 2 * x4 ** 3 + 9 * x4 ** 2 - 4)) ** 0.5

            int_s4 = integral(x4, y4)
            ans4.append(int_s4)
            ans.append(int_s4)

    fig, ((ax1)) = plt.subplots(1, 1, figsize=(12, 8))

    ax1.scatter(E1, ans1, label="ans1",  color='blue')
    ax1.scatter(E1, ans2, label="ans2",  color='red')
    ax1.scatter(E2, ans3, label="ans2", color='black')
    ax1.scatter(E3, ans4, label="ans2",  color='green')
    '''
    ax1.scatter(E0, ans, label="ans2", color='black')
    '''

    plt.show()

def periud(E):
    ar = []
    delt = 0.001
    prev = -100
    for x_ in np.linspace(-5, 5, 100000):
        if abs(3 * x_ ** 4 - 2 * x_ ** 3 - 9 * x_ ** 2 + 4 - E) < delt and abs(prev - x_) > 0.1:
            ar.append(x_)
            prev = x_
    print(len(ar))
    d1, u1 = ar[0], ar[1]
    d2, u2 = ar[2], ar[3]

    x = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
    y = 2 / (2 * (E - 3 * x ** 4 + 2 * x ** 3 + 9 * x ** 2 - 4)) ** 0.5

    x2 = np.linspace(d2 + 0.001, u2 - 0.001, 10000)
    y2 = 2 / (2 * (E - 3 * x2 ** 4 + 2 * x2 ** 3 + 9 * x2 ** 2 - 4)) ** 0.5

    print('Периуд левой ямы',integral(x, y))
    print()
    print('Периуд правой ямы', integral(x2, y2))

def qraf2():
    ans1 = []
    ans2 = []
    E0 = []

    E_ = np.linspace(-0.001, 4 , 100)

    for E in E_:
        ar = []
        delt = 0.001
        prev = -100
        for x_ in np.linspace(-5, 5, 100000):
            if abs(3 * x_ ** 4 - 2 * x_ ** 3 - 9 * x_ ** 2 + 4 - E) < delt and abs(prev - x_) > 0.1:
                ar.append(x_)
                prev = x_
        if len(ar) == 4:
            E0.append(E)
            d1, u1 = ar[0], ar[1]
            d2, u2 = ar[2], ar[3]

            x = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
            y = 2 / (2 * (E - 3 * x ** 4 + 2 * x ** 3 + 9 * x ** 2 - 4)) ** 0.5

            x2 = np.linspace(d2 + 0.001, u2 - 0.001, 10000)
            y2 = 2 / (2 * (E - 3 * x2 ** 4 + 2 * x2 ** 3 + 9 * x2 ** 2 - 4)) ** 0.5
            # print(d,u, len(ar))

            int_s1 = integral(x, y)
            int_s2 = integral(x2, y2)
            ans1.append(int_s1)
            ans2.append(int_s2)

    fig, ((ax1)) = plt.subplots(1, 1, figsize=(12, 8))

    ax1.scatter(E0, ans1, label="ans1",  color='blue')
    ax1.scatter(E0, ans2, label="ans2",  color='red')

    plt.show()

def qraf3():
    ans = []
    E = []

    E_ = np.linspace(3.99, 100 , 600)

    for E in E_:
        ar = []
        delt = 0.001
        prev = -100
        for x_ in np.linspace(-5, 5, 100000):
            if abs(3 * x_ ** 4 - 2 * x_ ** 3 - 9 * x_ ** 2 + 4 - E) < delt and abs(prev - x_) > 0.1:
                ar.append(x_)
                prev = x_

        if len(ar) == 2:
            E.append(E)
            d1, u1 = ar[0], ar[1]

            x3 = np.linspace(d1 + 0.001, u1 - 0.001, 10000)
            y3 = 2 / (2 * (E - 3 * x3 ** 4 + 2 * x3 ** 3 + 9 * x3 ** 2 - 4)) ** 0.5

            int_s3 = integral(x3, y3)
            ans.append(int_s3)


    fig, ((ax1)) = plt.subplots(1, 1, figsize=(12, 8))

    ax1.scatter(E, ans, label="ans1",  color='blue')

    plt.show()

E = 3

periud(E)

qraf2()
