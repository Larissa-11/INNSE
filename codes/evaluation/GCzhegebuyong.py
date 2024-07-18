import seaborn as sns
import matplotlib.pyplot as plt
from data import data_handle


def compute_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / len(sequence) * 100
    return gc_content


# 生成 DNA 序列并存储到不同文件中
encoding_methods = ['./code/DNAFountain.dna',
                    './code/Yin-Yang.dna', './code/DNA-QLC.dna', './code/INNSEM.dna']
gc_contents_all = []
total_sequences_all = []

for encoding_method in encoding_methods:
    dna_sequences = data_handle.read_dna_file(encoding_method)
    gc_contents = [compute_gc_content(sequence) for sequence in dna_sequences]
    gc_contents_all.append(gc_contents)
    total_sequences_all.append(len(dna_sequences))

# 设置样式
sns.set_style("white")

# 创建子图
fig, axs = plt.subplots(nrows=4, figsize=(10, 8), sharex=True, sharey=True)

for idx, ax in enumerate(axs):
    total_sequences = total_sequences_all[idx]

    sns.histplot(gc_contents_all[idx], kde=False, ax=ax, bins=5, color='#1f77b4', stat='percent')

    # 在左上角添加编码方式名称
    ax.text(0.03, 0.95, encoding_methods[idx].split('/')[-1].split('.')[0], fontsize=10, va='top', ha='left',
            transform=ax.transAxes)
# 共用横坐标
plt.xlabel('GC Content (%)')
# 调整外边距
plt.subplots_adjust(hspace=0.4)

# 保存图形
plt.savefig('GC_content0.png')

# 显示图形
plt.show()
