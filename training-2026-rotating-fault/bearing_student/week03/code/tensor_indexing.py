import torch
X = torch.arange(12).reshape(3,4)
print("原始矩阵 X：\n",X)

print("最后一个元素：",X [-1,-1])

print("切片结果：\n", X[1:3,1:])

X[0,0:2] = 5
print("修改后的 X：\n",X)