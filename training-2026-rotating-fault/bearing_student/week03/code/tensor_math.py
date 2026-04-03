import torch

# 1. 创建张量X 3行4列
X = torch.arange(12,dtype=torch.float32).reshape(3,4)
# 2.创建张量Y： 3行4列
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])

print("X的形状：\n",X.shape)
print("Y的形状：\n",Y.shape)
print("-"*50)

# 3. dim=0: 沿行/轴0 拼接 (上下拼在一起）
cat_dim0 = torch.cat((X , Y ), dim=0)
print("dim=0 上下拼接结果：\n" , cat_dim0)
print("拼接后形状：", cat_dim0.shape)
print("-"*50)

# 4. dim=1  沿列/轴1 拼接（左右贴在一起）
cat_dim1 = torch.cat((X , Y ), dim=1)
print("dim=1 左右拼接结果：\n",cat_dim1)
print("拼接后形状：", cat_dim1.shape)
print("-"*50)

print(X == Y )
#对张量中的所有元素进行求和，会产生一个单元素张量
print(X.sum())
print("-"*50)

# 广播机制
a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
#矩阵a将复制列，矩阵b将复制行，然后再按元素相加
print(a,b)
print(a+b)
print(a*b)
print("-"*50)

#索引和切片
#用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素：
print(X[-1],X[1:3])
#通过指定索引来将元素写入矩阵
X[0,2] = 9
print(X)
#为多个元素赋值相同的值:索引所有元素，然后为它们赋值
X[0:2,:] =12
print(X)
print("-"*50)


#节省内存
# 1. 记录Y原来的内存地址
before = id(Y)
print("Y原内存地址：",before )
# 2. 执行Y = Y + X
Y = Y + X
print("新地址与旧地址是否一致：",id(Y) == before)
print("Y现内存地址" ,id(Y))
print("-"*50)

Z = torch.zeros_like(Y)
print("修改前 id（Z）", id(Z))

Z[:] = X + Y
print("修改后的id（Z）", id(Z))

print("（Z）地址前后是否相同", id(Z) == before)
print("-"*50)