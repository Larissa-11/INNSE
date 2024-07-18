import seaborn as sns
import matplotlib.pyplot as plt

# 读取最小自由能数据
mfe_files = ['./MFE/DNAFountain.mfe', './MFE/Yin-Yang.mfe', './MFE/DNA-QLC.mfe', './MFE/INNSEM.mfe']
mfe_data = []

for file_path in mfe_files:
    print("Reading file:", file_path)  # 打印文件路径以便调试
    with open(file_path, 'r') as file:
        mfe_values = []
        for line in file.readlines():
            elements = line.strip().split()
            mfe_values.append(float(elements[0]))
        mfe_data.append(mfe_values)

# 设置样式
sns.set_style("white")

# 绘制直方图
fig, axs = plt.subplots(nrows=4, figsize=(8, 8), sharex=True, sharey=True)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, hspace=0.1)

for idx, ax in enumerate(axs.flatten()):
    mfe_values = mfe_data[idx]  # 获取对应编码方式的最小自由能数据
    sns.histplot(mfe_values, kde=False, ax=ax, stat='percent')
    # ax.set_title(mfe_files[idx].split('/')[-1].split('.')[0])
    ax.text(0.05, 0.9, mfe_files[idx].split('/')[-1].split('.')[0], transform=ax.transAxes, ha='left', va='top',
            fontsize='12')
    # ax.set_xlabel('Minimum Free Energy')
    # ax.set_ylabel('Frequency')
plt.xlabel('Minimum Free Energy (kJ/mol)')
# 添加整体标题
# plt.suptitle('Minimum Free Energy Distribution for Different Encoding Methods', fontsize=12)

# 显示图形
plt.show()

# 保存图形
plt.savefig('MFE_distribution.png')
