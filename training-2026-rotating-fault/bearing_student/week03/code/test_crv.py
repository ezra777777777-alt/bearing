
#导入操作系统工具 用来创建文件夹
import os

# 1. 创建一个文件夹  名叫data 放在上一级目录
# exist_ok = True 的意思：如果文件夹已经存在，不报错
os.makedirs(os.path.join('..','data'), exist_ok=True)

# 2. 指定要创建的文件路径： ../data/house_tiny.csv
data_file = os.path.join('..','data','house_tiny.csv')

# 3. 打开文件，开始写入内容
with open(data_file,'w') as f :
    f.write('NumRooms,Alley,Price\n')  #第一行：列表头（列的数据）
    f.write('NA,Pave,127500\n')        # 第一行的数据
    f.write('2,NA,106000\n')           # 第二行的数据
    f.write('4,NA,178100\n')           # 第三行的数据
    f.write('NA,NA,140000\n')          # 第四行的数据

# 4. 导入pandas数据处理库
import pandas as pd

# 5. 读取刚才创建的CSV文件
data = pd.read_csv(data_file)

# 6. 打印结果
print("这是我读取到的数据表格")
print(data)