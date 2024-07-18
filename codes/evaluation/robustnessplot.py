import csv
import numpy as np
import matplotlib.pyplot as plt

y_72=range(72)#生成0-31的数组
y_old=range(29)
row1=[]
with open('../results/Data_recording/robustness/DNA_Fountain_SNV.csv', encoding='UTF-8-sig') as f_DMC_erro_insert:
    for row in csv.reader(f_DMC_erro_insert, skipinitialspace=True):
        #print(row)#读出csv中的值
        row1.append(row)

row2 = sum(row1, [])
row3 = []
for r in row2:
    row3.append(float(r.rstrip('%')))


row11=[]
with open('../results/Data_recording/robustness/Yin_Yang_SNV.csv', encoding='UTF-8-sig') as f_DMC_erro_insert:
    for row in csv.reader(f_DMC_erro_insert, skipinitialspace=True):
        #print(row)#读出csv中的值
        row11.append(row)

row22 = sum(row11, [])
row33 = []
for r in row22:
    row33.append(float(r.rstrip('%')))

row111=[]
with open('../results/Data_recording/robustness/This_work_SNV.csv', encoding='UTF-8-sig') as f_DMC_erro_insert:
    for row in csv.reader(f_DMC_erro_insert, skipinitialspace=True):
        #print(row)#读出csv中的值
        row111.append(row)

row222 = sum(row111, [])
row333 = []
for r in row222:
    row333.append(float(r.rstrip('%')))
sum_list_72=[list(np.random.randint(8,12,12)/100000),list(np.random.randint(8,12,12)/10000),list(np.random.randint(20,60,12)/10000),list(np.random.randint(45,55,12)/10000),list(np.random.randint(75,85,12)/10000),list(np.random.randint(95,105,12)/10000)]
sum_list_180=[list(np.random.randint(8,12,30)/100000),list(np.random.randint(8,12,30)/10000),list(np.random.randint(20,60,25)/10000),list(np.random.randint(60,65,5)/10000),list(np.random.randint(45,55,25)/10000),list(np.random.randint(55,60,5)/10000),list(np.random.randint(75,85,30)/10000),list(np.random.randint(95,105,30)/10000)]
sum_list_72 = sum(sum_list_72, [])
sum_list_180 = sum(sum_list_180, [])
sum_list6=[0.0001, 0.001, 0.003, 0.005, 0.008, 0.01]
fig = plt.figure(figsize=(8, 6), facecolor='#FFFFFF')#facecolor是边框颜色

# plt.scatter(sum_list_180, row3, c='#00B8B8',#c是颜色 marker是形状 alpha是透明度
#            cmap='jet', edgecolors='black', linewidth=1, alpha=0.9, marker='o', label='DNA fountain')
#
# plt.scatter(sum_list_72, row33, c="#88c999",
#             cmap='jet', edgecolors='black', linewidth=1, alpha=0.9, s=100, marker='^', label='YYC')
#
# plt.scatter(sum_list_72, row333, c="#FC011A",
#             cmap='jet', edgecolors='black', linewidth=1, alpha=0.5, s=100, marker='*', label='FDMC')
plt.scatter(sum_list_180, row3, c='#eddd86', marker='o', alpha=0.9, label='DNA Fountain')  # 蓝色
plt.scatter(sum_list_72, row33, c="#efa666", marker='o', alpha=0.9,  label='Yin-Yang')  # 橙色
plt.scatter(sum_list_72, row333, c="#63b2ee", marker='o', alpha=0.9, label='INNSE')  # 绿色
plt.legend(loc='right', bbox_to_anchor=(1, 0.70), fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel('Sub error rate (%)', fontsize=15)#fontsize字的大小
plt.ylabel('% of binary data recovered', fontsize=15)


scale_ls = [0.0001, 0.001, 0.003, 0.005, 0.008, 0.01]

index_ls = ['0.01', '0.1', '0.3', '0.5', '0.8', '1']

plt.xticks(scale_ls,index_ls)
scale_y = [25, 50, 75, 100]

index_y = ['25', '50', '75', '100']

plt.yticks(scale_y,index_y)
plt.xlim(-0.0004,0.0104)

###画平均线
row_ave=[]
with open('../results/Data_recording/robustness/median_SNV.csv', encoding='UTF-8-sig') as f_DMC_erro_insert:
    for row in csv.reader(f_DMC_erro_insert, skipinitialspace=True):
        #print(row)#读出csv中的值
        row_ave.append(row)

row_ave_DNAF = row_ave[0]
row_ave_DNAF_num = []
for r in row_ave_DNAF:
    row_ave_DNAF_num.append(float(r.rstrip('%')))

row_ave_YYC = row_ave[1]
row_ave_YYC_num = []
for r in row_ave_YYC:
    row_ave_YYC_num.append(float(r.rstrip('%')))

row_ave_DMC = row_ave[2]
row_ave_DMC_num = []
for r in row_ave_DMC:
    row_ave_DMC_num.append(float(r.rstrip('%')))

plt.plot(sum_list6,row_ave_DNAF_num,color='#eddd86', marker='o',linestyle='-',alpha=0.9)
plt.plot(sum_list6,row_ave_YYC_num,color="#efa666", marker='o',linestyle='-',alpha=0.9)
plt.plot(sum_list6,row_ave_DMC_num,color="#63b2ee", marker='o',linestyle='-',alpha=0.9)
plt.savefig('robustness_SNV.png',dpi=100)
