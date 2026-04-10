# import numpy as np
# import matplotlib.pyplot as plt
#
# fs = 12000
# t_duration = 1.0
# t = np.arange(0.0, t_duration, 1/fs)
# print(t.shape)
#
# # --- 步骤 2 物理参数设定 ---
# fd = 50.0      # 故障特征频率 (Hz) - 假设滚子每秒磕碰剥落坑50次
# fn = 3000.0    # 结构固有频率 (Hz) - 轴承座被激发出的高频“嗡嗡”共振
# beta = 500.0   # 阻尼系数 - 决定共振衰减有多快（金属结构阻尼）
#
# # ==========================================
# # 你的任务：
# # 1. 计算两次冲击之间的绝对时间间隔（周期） T_d
# # 2. 计算局部时间 tau
# #    提示：让全局时间 t 对 T_d 取余数。体会一下机械凸轮每转一圈归零的直觉。
# # 3. 根据上面提供的衰减正弦波公式，计算出冲击信号 x
# #    提示：使用 np.exp() 和 np.sin()。Python 的广播机制会自动对整个 tau 数组进行计算。
# # ==========================================
#
# T_d = 1/fd
# tau = t % T_d   # 局部时间。每次 t 达到 0.02、0.04... 时，tau 就会瞬间变成 0
# resonance = np.sin(2 * np.pi * fn * tau)   # 高频振荡的弹簧
# envelope = np.exp(-beta * tau)
# x = envelope * resonance  # 衰减的包络线 压制了 高频的正弦波
#
# # ==========================================
# # 可视化环节
# # 避坑指南：如果我们画出全部 12000 个点，屏幕上会糊成一团。
# # 机械思维：我们先放大看看前 0.1 秒的细节，也就是前 1200 个点。
# # ==========================================
#
# plt.figure(figsize=(10, 3))
# plt.plot(t[:1200], x[:1200])  # 利用切片只画前 0.1 秒
# plt.title("Simulated Bearing Fault Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

############################################################

import numpy as np
import matplotlib.pyplot as plt
from pyexpat import features
from scipy.signal import hilbert
# ==========================================
# 步骤 1：定义物理参数 (图纸规范)
# 这一步就像在图纸标题栏写上：材料、公差、尺寸
# ==========================================
fs = 12000         # 采样率 (Hz)：每秒采集12000个点
t_duration = 1.0   # 总时长 (s)：我们要模拟1秒钟的信号
fd = 50.0          # 故障频率 (Hz)：每秒磕碰50次
fn = 3000.0        # 共振频率 (Hz)：金属结构的高频回音
beta = 500.0       # 阻尼系数：回音衰减的速度

# ==========================================
# 步骤 2：建立时间轴 (铺设流水线传送带)
# 没有时间，一切物理运动都不存在。必须先生成时间向量 t
# ==========================================
t = np.arange(0, t_duration, 1/fs)  # 从0到1秒，步长为1/12000

# ==========================================
# 步骤 3：核心机理组装 (制造零件并拼装)
# 在这里，我们把大问题拆成小零件，最后相乘或相加
# ==========================================
# 零件A：算出每次磕碰的周期
T_d = 1 / fd

# 零件B：制造一个不断归零的“局部秒表” (模拟反复磕碰)
tau = t % T_d

# 零件C：制造高频震动的弹簧 (里子)
resonance = np.sin(2 * np.pi * fn * tau)

# 零件D：制造减震的包络线 (外壳)
envelope = np.exp(-beta * tau)

# 总装配：把减震器套在弹簧上，得到最终的干净冲击信号
x_clean = envelope * resonance

# ==========================================
# 步骤 4：引入现实环境 (添加白噪声)
# 实验室里的信号不可能这么干净，我们要加上机器的轰鸣声
noise = np.random.normal(0, 0.5, len(t))
x_final = x_clean + noise
# ==========================================
# 暂时跳过，我们先确认干净的信号是对的
x_final = x_clean

# ==========================================
# 步骤 5：质检与可视化 (用眼睛收货)
# 像机械加工完要用三坐标测量仪打一下一样，我们必须画图确认
# ==========================================
plt.figure(figsize=(10, 3))
plt.plot(t[:1200], x_final[:1200])  # 只切取前0.1秒(1200个点)来看细节
plt.title("Simulated Bearing Fault Signal (Clean)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# ==========================================
# 步骤 6：特征提取 (FFT 三棱镜) - 新增！
# ==========================================
N = len(x_final) # 总点数 12000

# 1. 做 FFT 变换 (这一步算出来的是复数)
X_fft = np.fft.fft(x_final)

# 2. 计算出真实的振幅大小 (取模运算)
# 物理细节：需要除以 N 并乘以 2，才能还原真实的物理振幅
amplitude = np.abs(X_fft) / N * 2

# 3. 生成对应的频率 X 轴
# 频率最高只能看到采样率的一半（奈奎斯特定理：12000/2 = 6000 Hz）
freqs = np.fft.fftfreq(N, 1/fs)

# 为了画图好看，我们只取正频率部分（前一半）
half_N = N // 2
freqs_half = freqs[:half_N]
amplitude_half = amplitude[:half_N]

# ==========================================
# 可视化：看看频谱图长什么样
# ==========================================
plt.figure(figsize=(10, 4))
plt.plot(freqs_half, amplitude_half)
plt.title("FFT Spectrum of the Noisy Fault Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
# 我们把视野拉近，看看 0 到 4000 Hz 之间发生了什么
plt.xlim(0, 4000)
plt.grid(True)
plt.show()

# ==========================================
# 步骤 7：终极武器 - 包络分析 (Hilbert + FFT)
# ==========================================
# 1. 剥壳：用希尔伯特变换求解析信号，并取绝对值得到包络线
analytic_signal = hilbert(x_final)
envelope_signal = np.abs(analytic_signal)

# 🚨 导师避坑指南：去均值 (Remove DC Offset)
# 提取出的包络线全部在 X 轴上方（全为正数），这意味着它有一个很大的均值。
# 如果直接做 FFT，0 Hz 的位置会出现一个顶破天的巨大尖峰，把其他频率全压扁。
# 所以我们在做 FFT 前，必须把均值减掉，让信号重新跨越 X 轴上下波动。
envelope_signal = envelope_signal - np.mean(envelope_signal)

# 2. 对纯净的包络线做 FFT (拿着三棱镜照信封)
# 这里的 N, freqs_half 我们直接复用上一步算好的变量
envelope_fft = np.fft.fft(envelope_signal)
envelope_amp = np.abs(envelope_fft) / N * 2
envelope_amp_half = envelope_amp[:half_N]

# ==========================================
# 最终可视化：包络谱 (Envelope Spectrum)
# ==========================================
plt.figure(figsize=(10, 4))
# 我们用醒目的红色来画包络谱
plt.plot(freqs_half, envelope_amp_half, color='red')
plt.title("Envelope Spectrum (The Truth Revealed)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
# 机械思维：因为我们知道内鬼就在几十 Hz 附近，所以把显微镜对准 0 - 200 Hz
plt.xlim(0, 200)
plt.grid(True)
plt.show()
#============================================

import torch

#===========================================
# 步骤 8：特征提取与组装 (制作呈送给 MLP 的报表)
# ==========================================

# 1. 发出通缉令：明确我们想要提取的“内鬼”频率
# 在机械诊断中，故障特征频率的一倍频(1X)、二倍频(2X)、三倍频(3X)是最核心的特征
target_freqs = [50.0,100.0,150.0]
features = [] #准备一个空的手提箱，用来装提取出来的能量值

# 2. 拿着通缉令，去包络谱里抓人
for target in target_freqs:
    # 核心技巧：寻找最接近目标的门牌号 (Index)
    # np.abs(freqs_half - target) 会算出所有频率点距离目标的误差
    # np.argmin() 会返回误差最小的那个点的“数组索引”（第几个位置）
    idx = np.argmin(np.abs(freqs_half - target))

    # 根据门牌号，去振幅数组里把真实的能量值拿出来
    amp_value = envelope_amp_half[idx]

    #把拿到的能量值装进手提箱
    features.append(amp_value)

#打印出来看看我们提炼出来的纯金
print(f"提取出的纯净特征列表（Python List）：{features}")

# ==========================================
# 步骤 9：无缝对接上周的 PyTorch 模型
# ==========================================
# 将普通的 Python 列表，锻造成 PyTorch 认识的 Tensor 武器，并增加一个 Batch 维度
X_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

print(f"即将喂给 MLP 的张量形状: {X_input.shape}")
# 期待输出形状：torch.Size([1, 3])，代表 1 个样本，3 个高级特征