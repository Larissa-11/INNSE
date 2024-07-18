import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(8, 8), dpi=100)
plt.subplot(1, 1, 1)
N = 4
values = ( 513921, 360284, 0, 0)
index = np.arange(N)
width = 0.2
p2 = plt.bar(index, values, width)
plt.bar_label(p2, label_type='edge',fontsize=12)
plt.xlabel('Different encoding methods', fontsize=12, labelpad=10)
plt.ylabel('Number of undesired motifs', fontsize=12, labelpad=8.5)
# plt.title('The case of undesired motifs', fontsize=12, pad=20)
plt.xticks(index, ('DNA Fountain', 'Yin-Yang', 'DNA-QLC', 'INNSEM'))
plt.yticks(np.arange(0, 520000, 100000))
plt.subplots_adjust(left=0.19, right=0.9, top=0.9, bottom=0.1)
plt.savefig('Motifis.png',dpi=100)
plt.close()










