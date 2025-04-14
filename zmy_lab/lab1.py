import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 5 * x + 8 + torch.rand(x.size())
# 上面这行代码是制造出接近 y=5x+8 的数据集，后面加上 torch.rand() 函数制造噪音

"""画图 以下语句是显示散点图的，如果想看就把注释符号去掉，显示完散点图，需要关掉此图才能继续运行。"""
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出的维度都是 1

    def forward(self, x):
        out = self.linear(x)
        return out

if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
num_epochs = 1000

for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = x.cuda()
        target = y.cuda()
    else:
        inputs = x
        target = y

    # 向前传播
    out = model(inputs)
    loss = criterion(out, target)

    # 向后传播
    optimizer.zero_grad()  # 注意每次迭代都需要清零
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch[{epoch + 1}/{num_epochs}], loss:{loss.item():.6f}')

model.eval()
with torch.no_grad():
    if torch.cuda.is_available():
        predict = model(x.cuda())
        predict = predict.cpu().numpy()
    else:
        predict = model(x)
        predict = predict.numpy()

plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.legend()
plt.show()