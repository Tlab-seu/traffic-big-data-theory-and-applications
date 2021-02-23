#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
# 读取数据文件 
data = pd.read_csv('DATASET-B.csv')
# 转换数据类型
for c in ['rowid', 'colid', 'time_id']:  
    data[c] = data[c].astype(int)
# 排序
data = data.sort_values(['date', 'rowid', 'colid', 'time_id']).reset_index(drop=True)
data['date'] -= data['date'].min()


# In[4]:


def grid_recovery(df_, cols=[], lens=[]):  
    '''''修复缺失网格，填充0'''  
    df = df_.copy()  
    lcols = len(cols)  
    llens = len(lens)  
    if lcols != llens: # 确保输入的网格名称与网格长度信息的长度一致  
        raise ValueError(f'Lengths of cols ({lcols}) and lens ({llens}) mismatch.')  
    recovery_df = None
    for c, l in zip(cols, lens):  
        tmp_df = pd.DataFrame({c: range(l)}) # 完整网格  
        tmp_df['flag'] = True # 网格补全标记列 
        if recovery_df is None:  
            recovery_df = tmp_df.copy()  
        else:  
            recovery_df = recovery_df.merge(tmp_df, 'left', 'flag')  
    del recovery_df['flag']  
    df = recovery_df.merge(df, 'left', cols) # 补全所有网格  
    df = df.fillna(0) # 对缺失值补0  
    return df  
NROWS = 53 # 空间网格行数  
NCOLS = 67 # 空间网格列数  
NTIME = 144 # 时间网格数  
NDATE = 30 # 日期网格数
data = grid_recovery(data, ['date', 'rowid', 'colid', 'time_id'], [NDATE, NROWS, NCOLS, NTIME])  
for c in ['labels', 'volume', 'stopNum']:  
    data[c] = data[c].astype(int) # 调整数据类型  


# In[5]:


data['hourid'] = data['time_id'] // 6 # 合并时间网格  
data['new_rowid'] = data.rowid // 2 # 合并空间网格  
data['new_colid'] = data.colid // 2  
volume = data.groupby( # 计算合并网格后各网格的流量  
    ['date', 'new_rowid', 'new_colid', 'hourid']).volume.sum().reset_index()  
volume.columns = ['date', 'rowid', 'colid', 'hourid', 'volume']  


# In[6]:


data['seqid'] = data.date * 144 + data.time_id  
core_data = data.loc[    
    data.new_rowid.eq(10) & data.new_colid.eq(10)    
].reset_index(drop=True) # 提取待预测数据    
ts = core_data.groupby(['seqid']).agg({    
    'volume': 'sum',    
    'aveSpeed': 'mean',    
    'gridAcc': 'mean',    
    'speed_std': 'mean',    
    'hourid': 'mean',    
    'date': 'mean',    
    'time_id': 'mean'    
}).reset_index() # 统计各项特征，建立时间序列


# In[7]:


from sklearn.preprocessing import MinMaxScaler    
train_ts = ts[:-5*144].reset_index(drop=True)[[    
    'volume', 'aveSpeed', 'gridAcc', 'speed_std', 'hourid']] # 提取训练集序列    
test_ts = ts[-5*144:].reset_index(drop=True)[[    
    'volume', 'aveSpeed', 'gridAcc', 'speed_std', 'hourid']] # 提取测试集序列    
y_true_test = test_ts.values[..., 0]   # 测试集预测值真值    
scaler = MinMaxScaler()     # 初始化归一化工具    
scaler.fit(train_ts.values) # 读取数据最值信息    
train_ts = pd.DataFrame(scaler.transform(train_ts), columns=train_ts.columns) # 归一化训练集    
test_ts = pd.DataFrame(scaler.transform(test_ts), columns=test_ts.columns)    # 归一化测试集    


# In[8]:


import matplotlib.pyplot as plt  
import seaborn as sns  


# In[9]:


fig,ax = plt.subplots(4, 1, figsize=(10, 6), dpi=200, sharex=True)
for idx, (f, c) in enumerate(zip(['volume', 'aveSpeed', 'gridAcc', 'speed_std'], sns.color_palette('muted'))):
    sns.lineplot(x=ts[:-5*144].index, y=f, data=ts[:-5*144], color=c, ax=ax[idx])
    sns.lineplot(x=ts[-5*144:].index, y=f, data=ts[-5*144:], ax=ax[idx], color=c, alpha=0.2)
    ax[idx].grid()
    if idx == 2:
        ax[idx].set_ylim(-1, 2)
    elif idx == 3:
        ax[idx].set_ylim(-1, 11)


# In[ ]:





# In[10]:


import numpy as np  
from sklearn.model_selection import train_test_split  
# 将时间序列预测转换为监督学习问题. 
def ts_to_supervise(ts, window, forecast):  
    """ 
    ts: 原始时间序列 
    window: 输入时间步数量 
    forecast: 输入时间步数量 
    """  
    past = []   # 存储输入的时间序列值  
    future = [] # 存储输出的时间序列值  
    for i in range(len(ts) - window - forecast + 1):  
        past.append(ts[i:i+window])  
        future.append(ts[i+window:i+window+forecast])  
    return np.stack(past), np.stack(future)  
ts_x, ts_y = ts_to_supervise(pd.concat([train_ts, test_ts]).values, 7*144, 1)  
train_X, test_X, train_y, test_y = train_test_split(    
    ts_x, ts_y, test_size=len(test_ts), shuffle=False) # 测试集划分    


# In[11]:


import tensorflow as tf  
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional  
# 设定随机种子  
SEED = 233  
np.random.seed(SEED)  
tf.random.set_seed(SEED)  
# 循环神经网络模型
def rnn_model(input_shape):  
    inputs = Input(shape=input_shape)  
    x = LSTM(8)(inputs)  
    x = Dense(8)(x)  
    x = Dense(1)(x)  
    model = Model(inputs=inputs, outputs=x) # 建立模型  
    return model  
model = rnn_model(train_X.shape[1:]) # 实例化模型 


# In[12]:


from tensorflow.keras import optimizers  
from tensorflow.keras.callbacks import EarlyStopping  
batch_size = 16 # 每一训练批次的样本数量  
epochs = 50     # 最大训练轮数  
opt = optimizers.Adam(learning_rate=0.001) # 优化器  
model.compile(loss='mse',  
              optimizer=opt,  
              metrics=['mae', 'mse'])  
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20) # 早停策略  
history = model.fit(  
    train_X, train_y[..., 0],  
    batch_size=batch_size,  
    epochs=epochs,  
    validation_split=0.15, # 验证集比例  
    use_multiprocessing=True, # 使用多线程  
    callbacks=[early_stopping]) # 训练模型  


# In[13]:


from sklearn.metrics import mean_absolute_error, mean_squared_error  
pred_test = model.predict(test_X) # 预测测试集数据  
inv_test_ts = test_ts.copy()  
inv_test_ts.volume = pred_test  
y_hat_test = scaler.inverse_transform(inv_test_ts)[:, 0] # 还原归一化  
print(mean_absolute_error(y_true_test, y_hat_test),   
      mean_squared_error(y_true_test, y_hat_test),   
      np.sqrt(mean_squared_error(y_true_test, y_hat_test)))


# In[14]:


from matplotlib.font_manager import FontProperties
fp = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=11)


# In[63]:


fig,ax = plt.subplots(1, 1, figsize=(10, 2), dpi=200, sharex=True)
c = sns.color_palette('muted')
sns.lineplot(x=ts[:-5*144].index, y='volume', data=ts[:-5*144], color=c[0], ax=ax, label='真实值')
sns.lineplot(x=ts[-5*144:].index, y='volume', data=ts[-5*144:], ax=ax, color=c[0], alpha=0.3)
sns.lineplot(x=ts[-5*144:].index, y=y_hat_test, ax=ax, color=c[1], label='预测值')
ax.legend(prop=fp)
ax.set_xlim(-50, len(ts) + 50)
ax.set_ylabel('流量', fontproperties=fp)
ax.grid()
plt.show()

