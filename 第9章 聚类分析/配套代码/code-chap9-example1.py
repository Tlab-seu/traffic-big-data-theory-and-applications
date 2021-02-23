import numpy as np  
import pandas as pd  
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  
from sklearn.mixture import GaussianMixture  
from sklearn.preprocessing import StandardScaler  
# 读取数据  
data_ori = pd.read_csv('第十章数据.csv')  
# 选择特征  
feature = ['stopNum', 'aveSpeed']  
# 数据标准化  
scaler = StandardScaler()  
scaler.fit(data_ori[feature])  
data_ori_nor = scaler.transform(data_ori[feature])  
# K均值聚类  
n = 3  
labels = KMeans(n_clusters=n, random_state=0).fit(data_ori_nor).labels_  
# 输出数据集  
output_data = pd.concat((data_ori,   
                         pd.DataFrame(labels, columns = ['labels'])),   
                         axis=1)  
output_data.to_csv('kmeans聚类结果.csv', index=False)  
# 可视化
import pandas as pd  
import seaborn as sns  
from matplotlib import pyplot as plt  
import numpy as np  
plt.rcParams['axes.unicode_minus'] = False    
plt.rcParams['font.sans-serif'] = 'SimHei'  
# 读取数据  
df = pd.read_csv('kmeans聚类结果.csv')  
# 判断各簇实际意义  
grouped = df.groupby(['labels']).mean()['aveSpeed']  
congested = int(grouped.idxmin(axis=0))  
clear = int(grouped.idxmax(axis=0))  
slow = [x for x in [0,1,2] if x not in [congested, clear]][0]  
# 绘图  
fig = plt.figure(figsize=(8,6))  
plt.scatter(df[df['labels']==slow]['aveSpeed'], df[df['labels']==slow]['stopNum'], label='缓行', color='darkorange')  
plt.scatter(df[df['labels']==clear]['aveSpeed'], df[df['labels']==clear]['stopNum'], label='畅通', color='darkgreen')  
plt.scatter(df[df['labels']==congested]['aveSpeed'], df[df['labels']==congested]['stopNum'], label='拥堵', color='firebrick')  
plt.xlim((-0.1,40))  
plt.tick_params(labelsize=18)  
plt.xlabel('速度',fontsize=20)  
plt.ylabel('停车次数',fontsize=20)  
plt.legend(fontsize=18, loc='upper right') 
plt.savefig('k-means.png')