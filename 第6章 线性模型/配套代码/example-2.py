# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:35:04 2020

@author: user
"""

import numpy as np      
import pandas as pd    
import matplotlib.pyplot as plt      
from sklearn.datasets import make_blobs      
from sklearn.preprocessing import StandardScaler      
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LogisticRegression      
# 获取数据  
raw = pd.read_csv("DATASET-B.csv")    
input_data = raw[raw.date==20161101][['aveSpeed', 'stopNum', 'labels']]    
input_data.columns = ['平均速度', '停车次数', '拥堵程度']    
# 获取特征  
input = input_data.values    
X = input[:, :-1]    
y = input[:, -1]    
 # 对特征进行标准化      
nor_X = StandardScaler().fit_transform(X)        
X1_min, X1_max = nor_X[:,0].min(), nor_X[:,0].max()        
X2_min, X2_max = nor_X[:,1].min(), nor_X[:,1].max()         
# 划分训练集和测试集，比例 7:3      
X_train, X_test, y_train, y_test=train_test_split(nor_X, y, test_size=0.3, random_state=1)        
# 随机抽取训练集和测试集中的一些样本作图，较浅的点是测试集      
train_plot_n = 3000      
test_plot_n = 1000      
train_sample_plot_idx = np.random.choice (X_train.shape[0], size=train_plot_n, replace=False)      
test_sample_plot_idx = np.random.choice (X_test.shape[0], size=test_plot_n, replace=False)      
plt.scatter(X_train[train_sample_plot_idx][:,0],      
            X_train[train_sample_plot_idx][:,1],      
            c=y_train[train_sample_plot_idx],      
            edgecolors='k')        
plt.scatter(X_test[test_sample_plot_idx][:,0],      
            X_test[test_sample_plot_idx][:,1],      
            c=y_test[test_sample_plot_idx],      
            alpha=0.2, edgecolors='k')       
plt.xlabel('x1', fontsize=14)        
plt.ylabel('x2', fontsize=14)      
plt.xticks(fontsize=12)      
plt.yticks(fontsize=12)      
plt.xlim(X1_min-0.1, X1_max+0.1)        
plt.ylim(X2_min-0.1, X2_max+0.1)        
# plt.xlim(-1.5, 2)        
# plt.ylim(-0.5, 2.5)        
plt.savefig('./fig1.png', dpi=600)       
plt.show()   
# 训练LR模型      
logreg = LogisticRegression()        
logreg.fit(X_train, y_train)    
# 在 X1, X2 的范围内画一个 500*500 的方格，预测每个点的 label  
N,M = 500,500    
X1_min, X1_max = nor_X[:,0].min(), nor_X[:,0].max()    
X2_min, X2_max = nor_X[:,1].min(), nor_X[:,1].max()     
t1 = np.linspace(X1_min, X1_max, N)    
t2 = np.linspace(X2_min, X2_max, M)    
x1, x2 = np.meshgrid(t1,t2)    
x_star= np.stack((x1.flat, x2.flat),axis=1)    
y_star= logreg.predict(x_star)    
# 随机选取 sample_plot_n 个样本点  
sample_plot_n = 1000  
sample_plot_idx = np.random.choice(nor_X.shape[0], size=sample_plot_n, replace=False)  
plt.pcolormesh(x1,x2,y_star.reshape(x1.shape),alpha=0.1)    
plt.scatter(nor_X[sample_plot_idx][:,0],nor_X[sample_plot_idx][:,1],  
            c=y[sample_plot_idx]  
            ,edgecolors='k')    
plt.xlabel('x1', fontsize=14)    
plt.ylabel('x2', fontsize=14)  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
plt.xlim(X1_min-0.1, X1_max+0.1)    
plt.ylim(X2_min-0.1, X2_max+0.1)    
# plt.grid()    
plt.savefig('./fig2.png', dpi=600)  
plt.show()   
# 预测，计算准确率  
y_train_hat = logreg.predict(X_train)    
y_train = y_train.reshape(-1)    
result = y_train_hat == y_train    
c = np.count_nonzero(result)    
print('Train accuracy:%.2f%%'%(100*float(c)/float(len(result))))    
y_hat = logreg.predict(X_test)    
y_test = y_test.reshape(-1)    
result = y_hat == y_test    
c = np.count_nonzero(result)    
print('Test accuracy:%.2f%%'%(100*float(c)/float(len(result))))   
