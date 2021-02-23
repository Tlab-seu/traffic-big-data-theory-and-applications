# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:32:23 2020

@author: user
"""

import pandas as pd    
from sklearn.linear_model import LinearRegression    
import matplotlib.pyplot as plt    
raw = pd.read_csv("DATASET-B.csv")      # 读入数据  
feature = raw[(raw.rowid<30) & (raw.colid<30) & (raw.date==20161101)]     # 筛选数据  
s1 = feature[feature.time_id==47].set_index(['rowid', 'colid']).aveSpeed  # 时间网格为47的车速    
s2 = feature[feature.time_id==48].set_index(['rowid', 'colid']).aveSpeed  # 时间网格为48的车速    
s3 = feature[feature.time_id==49].set_index(['rowid', 'colid']).aveSpeed  # 时间网格为49的车速    
s4 = feature[feature.time_id==50].set_index(['rowid', 'colid']).aveSpeed  # 时间网格为50的车速    
data = pd.DataFrame(pd.concat((s1,s2,s3,s4), axis=1).values).dropna().reset_index(drop=True) # 拼接    
data.columns=['47','48','49','50'] # 修改列名    
X = data[['47', '48', '49']].values        
y = data['50'].values.reshape(-1, 1)        
reg = LinearRegression()     # 初始化        
reg.fit(X, y)     # 拟合数据        
coef = reg.coef_            # 自变量参数        
cons = reg.intercept_[0]      # 常数项        
print('变量参数：', coef, '常数项：', cons, 'R方：', reg.score(X, y))   
plt.rcParams['font.sans-serif']=['SimHei'] #设置中文字体    
plt.figure(figsize=(8, 6))     
plt.scatter(reg.predict(X), y, color='b')        
plt.plot([3, 18], [3, 18], "--", color='r', linewidth=4)  
plt.xlim(3, 18)  
plt.ylim(3, 18)  
plt.xticks(fontsize=14)    
plt.yticks(fontsize=14)    
plt.grid(linestyle='-.')    
plt.xlabel('预测值', fontsize=18)        
plt.ylabel('真实值', fontsize=18)  
plt.show()  
