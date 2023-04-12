# 导入numpy库
import numpy as np

# 读取csv文件，假设文件名为data.csv，分隔符为逗号
data = np.genfromtxt('test.csv', delimiter=',')

# 获取csv文件的列数
cols = data.shape[1]

# 遍历每一列，将其保存为npy文件，假设文件名为col_i.npy，其中i是列的索引
for i in range(cols):
    # 提取第i列的数据
    col = data[1:, i]
    # 保存为npy文件
    np.save(f'col_{i}.npy', col)
