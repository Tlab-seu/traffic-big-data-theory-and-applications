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
# 层次聚类  
n = 3  
labels = AgglomerativeClustering(n_clusters=n).fit(data_ori_nor).labels_  
# 输出数据集  
output_data = pd.concat((data_ori,   
                         pd.DataFrame(labels, columns = ['labels'])),   
                         axis=1)  
output_data.to_csv('层次聚类结果.csv', index=False)  