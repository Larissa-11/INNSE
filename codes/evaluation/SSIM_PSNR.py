import matplotlib.pyplot as plt
import numpy as np

# 数据
error_rates = ('0.1%', '0.2%', '0.5%', '1.0%', '1.5%', '2.0%')

# Grass_SSIM = (0.829, 0.54, 0.484, 0.328, 0.338, 0.304)
# Grass_PSNR = (37.117, 13.617, 10.717, 7.333, 6.101, 6.183)

YinYang_SSIM = (1, 0.642, 0.47, 0.379, 0.313, 0.322)
YinYang_PSNR = (100, 15.643, 10.144, 7.338, 7.969, 6.414)

DNAQLC_SSIM = (0.92, 0.821, 0.416, 0.57, 0.446, 0)
DNAQLC_PSNR = (36.24, 27.788, 17.495, 18.949, 14.469, 0)

This_work_SSIM = (0.89, 0.89, 0.89, 0.89, 0.89, 0.89)
This_work_PSNR = (33.98, 33.98, 33.98, 33.98, 33.657, 33.543)

# 调整柱状图位置
bar_width = 0.2
bar_positions = np.arange(len(error_rates))

# 绘制柱状图
fig, axes = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'wspace': 0.2})  # Adjust wspace here
# SSIM 柱状图
# axes[0].bar(bar_positions - 2*bar_width, Grass_SSIM, width=bar_width, label='Grass', align='center', color='darkkhaki', alpha=0.7)
axes[0].bar(bar_positions - bar_width, YinYang_SSIM, width=bar_width, label='YinYang', align='center', color='#efa666', alpha=0.7)
axes[0].bar(bar_positions, DNAQLC_SSIM, width=bar_width, label='DNA-QLC', align='center', color='#7cd6cf', alpha=0.7)
axes[0].bar(bar_positions + bar_width, This_work_SSIM, width=bar_width, label='This_work', align='center', color='#63b2ee', alpha=0.7)
axes[0].set_xlabel('Error Rate')
axes[0].set_ylabel('SSIM')
axes[0].set_xticks(bar_positions - 0.5*bar_width)  # Adjust x-ticks position
axes[0].set_xticklabels(error_rates)

# PSNR 柱状图
# axes[1].bar(bar_positions - 2*bar_width, Grass_PSNR, width=bar_width, label='Grass', align='center', color='darkkhaki', alpha=0.7)
axes[1].bar(bar_positions - bar_width, YinYang_PSNR, width=bar_width, label='YinYang', align='center', color='#efa666', alpha=0.7)
axes[1].bar(bar_positions, DNAQLC_PSNR, width=bar_width, label='DNA-QLC', align='center', color='#7cd6cf', alpha=0.7)
axes[1].bar(bar_positions + bar_width, This_work_PSNR, width=bar_width, label='This_work', align='center', color='#63b2ee', alpha=0.7)

axes[1].set_xlabel('Error Rate')
axes[1].set_ylabel('PSNR')
axes[1].set_xticks(bar_positions - 0.5*bar_width)  # Adjust x-ticks position
axes[1].set_xticklabels(error_rates)

# 添加图例
fig.legend(['Yin-Yang', 'DNA-QLC', 'INNSE'], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)

# 调整布局
plt.tight_layout(rect=[0, 0.1, 1, 1])

# 显示图表
plt.show()
plt.savefig('SSIM_PSNR_horizontal.png', dpi=100, bbox_inches='tight')
plt.close()
