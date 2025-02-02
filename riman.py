import geoopt
import torch

# целевая функция
def operation(v):
    return sum(v**2)

# параметры модели
v = torch.rand(5, requires_grad=True)

# инициализация оптимизатора
optimizer = geoopt.optim.RiemannianAdam(params=[v])

# функция замыкания
def closure():
    optimizer.zero_grad()
    loss = operation(v)
    loss.backward()
    return loss

# цикл оптимизации
for i in range(1000):
    optimizer.step(closure)

# отображение оптимизированного параметра
print(v)
