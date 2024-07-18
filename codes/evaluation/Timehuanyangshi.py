import matplotlib.pyplot as plt
import numpy as np

image_sizes_kb = np.array([124, 518, 1188, 4598])

encoding_times_DNAFountain = np.array([2.97, 22.92, 43.16, 190.92])
decoding_times_DNAFountain = np.array([1.51, 27.62, 42.91, 492.21])

encoding_times_YinYangCode = np.array([214.71, 12.50, 64.42, 95.35])
decoding_times_YinYangCode = np.array([0.44, 1.20, 1.95, 8.76])

encoding_times_DNAQLC = np.array([1.49, 36.44, 49.32, 133.63])
decoding_times_DNAQLC = np.array([2.43, 69.10, 92.83, 243.85])

encoding_times_Thiswork = np.array([3.14, 3.17, 3.39, 4.52])
decoding_times_Thiswork = np.array([2.95, 3.15, 4.07, 13.86])

# 创建两个子图，都是柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5.2))

# 调整柱状图的位置和间距 - 编码
bar_width = 0.2
index = np.arange(len(image_sizes_kb))

# 计算每组数据的中心位置
index_grouped = index - bar_width / 2

ax1.bar(index_grouped - bar_width, encoding_times_DNAFountain, width=bar_width, color='#eddd86', alpha=0.8)
ax1.bar(index_grouped, encoding_times_YinYangCode, width=bar_width, color='#efa666', alpha=0.8)
ax1.bar(index_grouped + bar_width, encoding_times_DNAQLC, width=bar_width, color='#7cd6cf', alpha=0.8)
ax1.bar(index_grouped + bar_width*2, encoding_times_Thiswork, width=bar_width, color='#63b2ee', alpha=0.8)

ax1.set_xlabel('Image Sizes (KB)')
ax1.set_yscale('log')
ax1.set_ylabel('Encoding Time (s)')
ax1.set_xticks(index)
ax1.set_xticklabels([f"{size}" for size in image_sizes_kb])

# 曲线图表示解码时间
ax2.bar(index_grouped - bar_width, decoding_times_DNAFountain, width=bar_width, color='#eddd86', alpha=0.8)
ax2.bar(index_grouped, decoding_times_YinYangCode, width=bar_width, color='#efa666', alpha=0.8)
ax2.bar(index_grouped + bar_width, decoding_times_DNAQLC, width=bar_width, color='#7cd6cf', alpha=0.8)
ax2.bar(index_grouped + bar_width*2, decoding_times_Thiswork, width=bar_width, color='#63b2ee', alpha=0.8)
ax2.set_yscale('log')  # 对解码时间进行对数变换
ax2.set_xlabel('Image Sizes (KB)')
ax2.set_ylabel('Decoding Time (s)')
ax2.set_xticks(index)
ax2.set_xticklabels([f"{size}" for size in image_sizes_kb])
for ax in [ax1, ax2]:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
# ax2.legend(['DNA Fountain', 'Yin-Yang', 'DNA-QLC', 'INNSE'], loc='upper left', fancybox=True, shadow=True, ncol=1, fontsize=10)
fig.legend(['DNA Fountain', 'Yin-Yang', 'DNA-QLC', 'INNSE'], loc='lower center', fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.001), ncol=4, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 1])  # 调整布局，防止重叠
plt.show()
plt.savefig('Time.png', dpi=300)
