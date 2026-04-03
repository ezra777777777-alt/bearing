from importlib.metadata import requires

import torch
from sympy.codegen.fnodes import reshape

from bearing_student.week03.code.tensor_math import before

#先创建X 和 Y （沿用之前的3行4列的张量）
# X = torch.arange (12,dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
#
# #1. 创建和 Y 形状完全相同的全0张量 Z
# Z = torch.zeros_like(Y)
# print('修改前 id（Z）：',id(Z))   #打印原始内存
#
# #2. 原地操作：切片赋值，不换内存
# Z[:] = X + Y
# print('修改后 id（Z）：',id(Z))   #打印修改后的地址
#
# #3. 验证地址是否相同
# print("地址是否相同？", id(Z) == id(Z))
#
# print("-"*50)
#
# before = id(X)
# print("X原id：",before)
#
# X += Y  #原地操作： X += Y （等价于 X[:] = X + Y ）
#
# #验证地址是否相同
# print("X修改后id：",id(X))
# print("地址是否相同：",id(X) == before)
# print("-"*50)
#
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A),type(B))
# print(A,B)
# print("-"*50)
#
# a = torch.tensor([3.5])
# print("张量 a:", a)
# print("a.item():", a.item())    # 转 Python 标量（推荐）
# print("float(a):", float(a))      # 转 Python 浮点数
# print("int(a):", int(a))          # 转 Python 整数（会截断小数）
# print("-"*50)

#练习
#1. 运行本节中的代码。将本节中的条件语句X == Y更改为X < Y或X > Y，然后看看你可以得到什么样的张量。
# X = torch.arange(12,dtype=torch.float32).reshape(3,4)
# Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
# print( X > Y )

#2. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？
# a = torch.arange(24).reshape((2,3,4))
# print(a)
# b = torch.arange(12).reshape((3,4))
# print(b)
# c = a + b
# print(c)
# print("-"*50)

# 1. 創建一個張量，並告訴 PyTorch：這個變量需要計算梯度 (requires_grad=True)
x = torch.arange(4.0,requires_grad=True)
print("x:",x)
print("-"*30)

# 2. 定义函數 y = 2 * (x的平方和)
y = 2 * torch.dot(x,x)
print("y:",y)
print("-"*30)

# 3. 反向传播（求导核心指令）
y.backward()

# 4. 查看 x 的梯度（即 dy/dx)
# 预期结果应该是 4*x = [0,4,8,12]
print("x 的梯度 (dy/dx)",x.grad)
print("-"*30)

#验证
print("验证是否正确：",x.grad == 4 * x)


x = torch.arange(4.0,requires_grad=True)
print("初始x：",x)

# ------------------ 核心操作开始 ------------------
if x.grad is not None:
    x.grad.zero_()  # 1. 先把之前的梯度清零（必做！避免累加干扰）
y = x * x       # 2. 定义 y（y 和 x 有关，在计算图里）
u = y.detach()# 3. 核心！把 y 从计算图里“剪下来”，u 继承值但不带梯度
z = u * x       # 4. 用 u（常数）和 x（变量）计算 z
# ------------------ 核心操作结束 ------------------
print("z 的 requires_grad 状态:", z.requires_grad)
# 反向传播求梯度
z.retain_grad()
z.sum().backward()

# 查看结果
print("y的值：",y)
print("u的值：",u)        # u 和 y 的值完全一样
print("u 是否有梯度追踪：", u.requires_grad)    # 会输出 False
print("x 的梯度：", x.grad)    # 这里是关键！
print("z 的值:", z)
print("z 的梯度:", z.grad)
print("~"*50)

def f (a):
    b = a * 2
    print("初始b ：",b)
    while b.norm() < 1000:
        b = b * 2
        print("循环中：",b )
    print("循环结束后 b：", b )

    if b.sum() > 0:
        c = b
        print("分支：c = b ")
    else:
        c = 100 * b
        print("分支：c = 100 * b")
    return c, b #同时返回b ，方便计算K

#初始化 a （为了方便验证，我们可以手动设一个a，比如 a=2)
a = torch.tensor(2.0,requires_grad=True)   #手动设a = 2.0(正数）
d,b = f(a)
d.backward()

#计算常数 k = d / a
k = d / a
print("\na的值：",a )
print("d 的值：", d )
print("常数 k = d/a:",k.item())
print("a 的梯度：", a.grad)
print("验证：a.grad == k?",torch.allclose(a.grad,k))