import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. 模拟1000个样本， 每个样本有10个特征值（假设是我们提取的频域特征）
X = torch.randn(1000,10)
#模拟这1000个样本的标签：0（正常），1（内圈故障），2（外圈故障）
y = torch.randint(0,3,(1000,))

# 2. 将X和y打包成一个Dataset
dataset = TensorDataset(X,y)


#交给Dataloader管理，设置每个批次32个样本，并且打乱顺序（shuffle=True）
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

# 4. 测试抽取一个批次
for batch_X,batch_y in dataloader:
    print("一个批次的X形状：", batch_X.shape)
    print("一个批次的y的形状：", batch_y.shape)

    break  #演示只看第一个批次即可