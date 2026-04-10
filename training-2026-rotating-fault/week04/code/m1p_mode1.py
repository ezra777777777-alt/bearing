import torch
from torch import nn

# 定义一个继承自 nn.Module 的网络类
class FaultDiagnosisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 假设输入特征是 10 维，隐藏层放 64 个神经元，最后输出 3 种故障概率
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),           # 加入非线性激活函数
            nn.Linear(64, 3)     # 最终输出 3 个类别的得分
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# 实例化模型
model = FaultDiagnosisMLP()
print(model)

###################################################
# 假设你用传感器采集了一段连续的轴承振动信号，它在 PyTorch 中是一个一维张量，长度为 1024：
# signal = torch.randn(1024)
# 问题：请写出一行代码，将这个 signal 转换成一个二维矩阵，要求包含 16 个批次（样本），每个样本包含 64 个数据点。
# import torch
#
# signal = torch.randn(1024)
# reshaped_signal = signal.reshape(16,64)
# print(reshaped_signal)
########################################################
import torch

from bearing_student.week03.code.tensor_math import before

# A = torch.ones(32, 10)  # 代表 32 个样本，每个样本 10 个特征
# B = torch.arange(10)    # 代表 10 个特征的补偿系数
# C = A + B
# #代码运行后，张量 C 的 shape 是多少？这在物理意义上相当于对数据做了什么操作？
# print("张量C的shape是：",C.size())
# print("矩阵B复制行 矩阵A复制列 再元素相加")
########################################################

# 在处理几个 G 的 CWRU 轴承数据集时，内存很宝贵。
# 问题：要把张量 X 的值加上 Y，写成 X = X + Y 和写成 X += Y，在底层内存分配上有什么本质区别？
# X = 12
# Y = 6
# before = id(X)
# X = X + Y
# print(id(X)==before)
#
# X += Y
# print(X)
# print("X + Y 内存分配上分配了新内存 创建了新张量 改变了内存地址 保留了原始数据  X += Y 反之 ")

###########################################################

#在写训练循环时，我们有一句极其关键的代码：
# optimizer.zero_grad()（或者 w.grad.zero_()
# ）。
# 问题：如果你在代码中删掉了这一行，模型继续训练下去会发生什么灾难性的后果？为什么？
# print("梯度未清零 误差值越来越大 偏离预设值")

###########################################################
#假设你提取了故障轴承的频域特征，总共有 10050 个训练样本。你使用了 DataLoader：
# from torch import nn
# from torch.utils.data import dataset, DataLoader
#
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 在一个 Epoch（一轮）的训练中，
# for batch_X, batch_y in dataloader:
#     这个循环会执行多少次？最后一个 Batch 里会有多少个样本？
# print("循环会执行315次   最后一次循环的样本只有两个")

###############################################################

# 假设你现在要对轴承进行 4 种状态的分类（正常、内圈故障、外圈故障、滚动体故障）。
# 你提取了信号的 12 个时域特征（如均方根、峰值、峭度等）作为输入。
# 问题：请用 nn.Sequential 补全下面的网络，要求：
#
# 第一层将 12 维特征映射到 32 维。
#
# 加上一个 ReLU 激活函数。
#
# 第二层将 32 维映射到最终的类别数。

class BearingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
          nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,4)# ___ 请在这里填入三行代码 ___
        )
    def forward(self, x):
        return self.net(x)

# ===================== 核心：输出预测结果 =====================
# 1. 实例化模型
model = BearingClassifier()

# 2. 构造输入：12个时域特征（模拟真实数据）
# 形状必须是 [批量大小, 12]，这里用1个样本测试：shape=(1,12)
input_features = torch.randn(1, 12)

# 3. 前向传播，得到网络原始输出（logits）
output = model(input_features)

# 4. 输出1：打印原始分类分数（未归一化）
print("网络原始输出 (4个类别的分数)：")
print(output)
print("输出形状：", output.shape)  # torch.Size([1, 4])

# 5. 输出2：转为概率（Softmax归一化，总和=1）
prob = torch.softmax(output, dim=1)
print("\n每个状态的概率：")
print(prob)

# 6. 输出3：最终预测类别（取概率最大的类别）
pred_class = torch.argmax(prob, dim=1).item()
# 定义轴承4种状态
status = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
print(f"\n最终预测结果：{status[pred_class]}")

#如果在上面的代码中，我们把 nn.ReLU() 删掉，
# 只保留两个 nn.Linear 层，这个网络会退化成什么？为什么我们必须加非线性激活函数？
#  无 ReLU：网络只能学简单线性关系，故障分类精度极低；
#  有 ReLU：网络能挖掘特征的复杂规律，精准区分正常、内圈、外圈、滚动体故障。

##############################################################

