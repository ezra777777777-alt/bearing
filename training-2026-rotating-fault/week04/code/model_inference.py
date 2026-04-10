import torch
from torch import nn

# 1. 声明模型结构
class BearingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6,16),
            nn.ReLU(),
            nn.Linear(16,3),
        )
    def forward(self,x):
        return self.backbone(x)

# 2. 实例化模型并加载
model = BearingClassifier()

# 3. 模拟“闭卷考试”数据
X_test = torch.randn(20,6)
y_test = torch.randint(0,3,(20,))

# 开始推理
model.eval() #评估模式
with torch.no_grad():
    outputs = model(X_test)
    #取概率最大的那一项作为预测的故障类型
    _, predicted = torch.max(outputs, 1)

# 4. 打印对比结果
print(f"预测结果：{predicted}")
print(f"真实标签：{y_test}")