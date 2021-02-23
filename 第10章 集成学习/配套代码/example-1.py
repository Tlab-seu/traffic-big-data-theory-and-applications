#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd    
# 读取数据文件   
data = pd.read_csv("DATASET-B.csv")
# 转换数据类型  
for c in ['rowid', 'colid', 'time_id']:  
    data[c] = data[c].astype(int) 


# In[2]:


# 排序 
data = data.sort_values(['date', 'rowid', 'colid', 'time_id']).reset_index(drop=True)  
data['datetime'] = pd.to_datetime(data.date, format='%Y%m%d')  #转换日期格式  
data['dayofweek'] = data.datetime.dt.dayofweek #提取星期信息   
data = data.sample(50000, random_state=233) # 随机选取50000条数据


# In[3]:


from sklearn.model_selection import train_test_split
features = [
    'rowid', 'colid', 'time_id', 'dayofweek',
    'aveSpeed', 'gridAcc', 'volume', 'speed_std', 'stopNum'
]
train, test = train_test_split(data, test_size=0.3, random_state=233)  #数据集划分 


# In[4]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
rf = RandomForestClassifier(  
    n_estimators=256,      #森林中的决策树数量  
    max_depth=9,         #决策树的最大深度  
    min_samples_leaf=30,   #叶子节点包含的最小样本数量  
    n_jobs=-1,            #模型拟合使用的处理器数量  
    random_state=233      #随机种子，用于获得可复制的结果  
)  #随机森林模型  
gbdt = GradientBoostingClassifier(  
    learning_rate=0.05,   #选择模型类型  
    n_estimators=256,     #迭代轮数  
    max_depth=8,        #决策树的最大深度  
    subsample=0.8,       #数据采样比例  
    max_features=0.9,  #特征采样比例  
    min_samples_split=5,   #划分叶子节点所需的最小样本数量  
    min_samples_leaf=30,        #叶子节点包含的最小样本数量 
    random_state=233     #随机种子，用于获得可复制的结果  
)  #梯度提升树模型  


# In[5]:


'''''随机森林训练'''  
rf.fit(train[features], train['labels'])  
rf_train_score = rf.score(train[features], train['labels']) #计算训练集精度  
rf_test_score = rf.score(test[features], test['labels'])   #计算测试集精度  
print(f'随机森林模型精度：训练集：{rf_train_score:.3f}；测试集：{rf_test_score:.3f}')  


# In[6]:


'''''梯度提升树训练'''  
gbdt.fit(train[features], train['labels'])  
gbdt_train_score = gbdt.score(train[features], train['labels']) #计算训练集精度  
gbdt_test_score = gbdt.score(test[features], test['labels'])   #计算测试集精度  
print(f'GBDT模型精度：训练集：{gbdt_train_score:.3f}；测试集：{gbdt_test_score:.3f}') 


# In[7]:


from sklearn.model_selection import learning_curve
import numpy as np
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,  
                    n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):  
    '''''训练曲线绘制函数'''  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel("Training examples")  
    plt.ylabel("Score")  
    train_sizes, train_scores, test_scores = learning_curve(  
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) #获取训练曲线  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()  
    # 绘制曲线
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                 train_scores_mean + train_scores_std, alpha=0.1,  
                 color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",  
          label="Training score")  
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",  
          label="Validation score")  
    plt.legend(loc="lower right")  
    return plt


# In[8]:


import matplotlib.pyplot as plt
# %matplotlib inline


# In[9]:


plot_learning_curve(rf, "Random Forest", train[features], train['labels'], cv=5, n_jobs=-1)  
plt.show() #绘制随机森林训练曲线


# In[10]:


plot_learning_curve(gbdt, "GBDT", train[features], train['labels'], cv=5, n_jobs=-1)  
plt.show() #绘制梯度提升树训练曲线 
