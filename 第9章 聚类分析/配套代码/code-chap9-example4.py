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
# DBSCAN  
eps = 0.5  
min_samples = 3  
labels = DBSCAN(eps=eps, min_samples=min_samples).fit(data_ori_nor).labels_ 
# 输出数据集  
output_data = pd.concat((data_ori,   
                         pd.DataFrame(labels, columns = ['labels'])),   
                         axis=1)  
output_data.to_csv('DBSCAN聚类结果.csv', index=False)  