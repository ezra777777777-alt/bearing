import torch
import torch.nn as nn

extracted_features = [1.5,0.8,0.3]

# 将 Python 列表转为 PyTorch 张量，并加上 Batch 维度 (变成 [1, 3])
X_input = torch.tensor(extracted_features, dtype=torch.float32).unsqueeze(0)
print(f"送入流水线的零件尺寸 (Input Shape): {X_input.shape}")

# ==========================================
# 2. 搭建质检流水线 (全连接神经网络)
# ==========================================
class SimpleBearingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的任务：请把下面的流水线补充完整！
        # 提示 1：第一层（前线汇报员 -> 隐藏分析师）。输入特征是 3，假设我们请了 8 个资深分析师（隐藏节点设为 8）。
        # 提示 2：加装一个单向阀（激活函数 ReLU）。
        # 提示 3：第二层（隐藏分析师 -> 厂长拍板）。将 8 个分析师的意见，汇总给 3 个厂长（输出 3 个分类状态：正常/内圈/外圈）。
        self.pipeline = nn.Sequential(
            nn.Linear(in_features=3, out_features=8),  # 请填入正确的数字
            nn.ReLU(),                                 # 单向溢流阀
            nn.Linear(in_features=8, out_features=3)   # 请填入正确的数字
        )

    def forward(self, x):
        return self.pipeline(x)

    # ==========================================
    # 3. 试车：执行前向传播
    # ==========================================
    # 实例化机床
model = SimpleBearingMLP( )

    # 你的任务：把材料 X_input 喂给 model，并把结果赋值给变量 output
output = model(X_input)

print(f"流水线吐出的原始打分 (Output Tensor): \n{output}")
print(f"打分表的尺寸 (Output Shape): {output.shape}")