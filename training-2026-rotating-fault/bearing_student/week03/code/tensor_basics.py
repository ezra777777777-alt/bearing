import torch
x = torch.arange(12)
print("张量 shape: ", x.shape)

print("张量内容：", x)

print("元素总数：", x.numel() )

X = x.reshape(3,4)
print("3*4矩阵：", X )

torch.zeros((2,3,4))
print("元素为0张量：",torch.zeros(2,3,4) )

torch.ones((2,3,4))
print("元素为1张量：",torch.ones(2,3,4))

x =torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y,x-y,x*y,x/y, x ** y)


torch.exp(x)
print(torch.exp(x))