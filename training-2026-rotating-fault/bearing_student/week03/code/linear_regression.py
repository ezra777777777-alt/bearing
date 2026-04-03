import torch

# 1. 准备数据：假设真实的 w=2.0 b=0.5
X = torch.randn(100,1) * 10
y_true = -1.5 * X + 3.0 + torch.randn(100,1) * 0.1 # 加点噪声模拟实际测量

# 2. 初始化模型参数（随机给个起点）
w = torch.randn(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# 3. 学习循环
learning_rate = 0.005
for epoch in range(300):
    #前向传播：预测
    y_pred = X * w + b

    #计算损失（误差平方和的平均值）
    loss = ((y_pred - y_true) ** 2).mean()

    #反向传播：计算梯度
    loss.backward()

    #更新参数：让 w 和 b 往误差减小的方向挪一小步
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        #记得清零梯度，迎接下一轮计算
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1 ) % 20 ==0 :
        print(f"Epoch {epoch + 1}: Loss {loss.item():.4f},w: {w.item():.2f}, b: {b.item():.2f}")


# 练习一：张量变形记（数据预处理基础）
# 在处理传感器采集的数据时，我们经常需要改变数据的维度结构。
#
# 任务：
# 假设你用采集卡收集到了一段包含 24 个数据点的连续振动信号，目前在 PyTorch 中它表现为一个一维长条：

# 请用 reshape 操作，把这个 signal 变成 3 个批次（Batch），每个批次包含 8 个数据点（即一个 3 行 8 列的矩阵）。
#
# 将变形后的结果打印出来，并使用 .shape 验证它的形状是否为 [3, 8]。

# import  torch
#
# signal = torch.randn(24)
#
# reshaped_signal = signal.reshape(3,8)
# print("原形状",reshaped_signal)
# print("Reshape 后形状", reshaped_signal.shape)


# 练习二：失控的学习率（体会梯度爆炸）
# 机器学习中，调整参数（调参）是家常便饭。学习率（Learning Rate）决定了模型每次纠错时“迈出的步子”有多大。
#
# 任务：
# 打开你刚才写好的 linear_regression.py 代码。
#
# 找到这一行：learning_rate = 0.001
#
# 将它修改为一个非常大的值，比如：learning_rate = 0.5
#
# 重新运行程序，观察控制台打印的 Loss 值。你会看到数字变成了 nan（Not a Number）或者无穷大。这就是经典的“步子迈太大，直接跨过了正确答案所在的坑，最后彻底迷失方向”的现象。


# 🎯 练习三：更换传感器标定（修改底层规律）现在我们要检验模型是否真的具备“学习”能力，而不是瞎猫碰上死耗子。
# 任务：假设我们换了一个新批次的传感器，它内部的物理规律变了。真实的系数不再是 $2.0$ 和 $0.5$，而是变成了 $w = -1.5$ 和 $b = 3.0$。
# 在 linear_regression.py 中，找到制造“实验数据”的那行代码，将其修改为新的规律：
# Pythony_true = -1.5 * X + 3.0 + torch.randn(100, 1) * 0.1 # 稍微减小了一点噪声方便观察
# 不要修改机器初始化的盲猜参数（依旧让 w 和 b 随机初始化或设为 0）。
# 将学习率改回正常的 0.001。
# 运行程序，观察在 100 轮训练结束后，机器自己算出来的 w 和 b 是否成功逼近了 -1.5 和 3.0。如果不够接近，尝试把循环次数从 100 改成 300 再次运行。
