#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd  
# 读取数据文件 
data = pd.read_csv('DATASET-B.csv')
# 转换数据类型
for c in ['rowid', 'colid', 'time_id']:  
    data[c] = data[c].astype(int)
# 排序
data = data.sort_values(['date', 'rowid', 'colid', 'time_id']).reset_index(drop=True)
data['date'] -= data['date'].min()


# In[6]:


print(len(data[['rowid', 'colid']].drop_duplicates()), # 数据集中包含空间网格数量
      (data.rowid.max() + 1) * (data.colid.max() + 1)) # 空间网格总数
# 输出：2082 3551
print(data.groupby(['rowid', 'colid']).time_id.nunique().mean(), # 数据集中各空间网格平均包含的时间段数量
      data.time_id.max() + 1) # 时间段总数
# 输出：108.15658021133525 144


# In[4]:


import matplotlib.pyplot as plt  
import seaborn as sns  
plt.figure(dpi=300)  
sns.heatmap(data.groupby(['rowid', 'colid']).size().reset_index(  
    name='volume').pivot_table('volume', 'rowid', 'colid'), cmap='Blues')
plt.show()  


# In[5]:


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


# In[6]:


data['hourid'] = data['time_id'] // 6 # 合并时间网格  
data['new_rowid'] = data.rowid // 2 # 合并空间网格  
data['new_colid'] = data.colid // 2  
volume = data.groupby( # 计算合并网格后各网格的流量  
    ['date', 'new_rowid', 'new_colid', 'hourid']).volume.sum().reset_index()  
volume.columns = ['date', 'rowid', 'colid', 'hourid', 'volume']  


# In[7]:


volume_pivot = volume.pivot_table(  
    index=['date', 'hourid', 'rowid'],  
    columns='colid',  
    values='volume').reset_index() # 网格转换  
volume_pivot['timeseq'] = volume_pivot['date'] * 24 + volume_pivot['hourid'] # 时间序号  
volume_pivot_np = volume_pivot[[c for c in range(34)]].values # 提取流量数值


# In[8]:


import numpy as np
def gen_movie(df, nrows=27, ncols=34, ntime=30*24, win_s=10, win_t=3):  
    n_i = nrows - win_s + 1   # 18
    n_j = ncols - win_s + 1   # 25
    piece = []  
    for t in range(24, ntime):  
        for i in range(n_i):  
            for j in range(n_j):  
                # 周期特征  
                prd_piece = df[t - 24:t - 23, i:i+win_s, j:j+win_s]  
                # 邻近特征  
                nbr_piece = df[t-win_t:t+1, i:i+win_s, j:j+win_s]  
                piece.append(np.vstack([prd_piece, nbr_piece]))  
    return np.stack(piece)  
movie = gen_movie(np.asarray(volume_pivot_np.reshape((30*24, 27, 34)), order='C'))  
np.save('data_x.npy', movie[:, :4])  
np.save('data_y.npy', movie[:, 4])  


# In[ ]:





# In[16]:


import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras.layers import Conv2D, BatchNormalization  
from tensorflow.keras.layers import Activation, MaxPooling2D  
from tensorflow.keras.regularizers import l2
# 卷积层
def conv_layer(inputs,  
             num_filters=16,  
             kernel_size=3,  
             strides=1,  
             data_format='channels_first',  
             activation='relu',  
             batch_normalization=True,  
             maxpooling=True,  
             pool_size=2,  
             pool_strides=2):  
    conv = Conv2D(num_filters,  
                 kernel_size=kernel_size,  
                 strides=strides,  
                 padding='same',  
                 data_format=data_format,  
                 kernel_regularizer=l2(1e-4))  
    x = conv(inputs)  # 卷积
    if batch_normalization:
        x = BatchNormalization()(x) # 批归一化
    if activation is not None:
        x = Activation(activation)(x) # 激活函数
    if maxpooling:
        x = MaxPooling2D(pool_size=pool_size,  
                        strides=pool_strides,  
                        data_format=data_format,  
                        padding='same')(x) # 池化
    return x


# In[17]:


from tensorflow.keras.layers import UpSampling2D  
def upsample_layer(inputs,  
                up_size=2,  
                interpolation='nearest',  
                data_format='channels_first'):
    upsample = UpSampling2D(size=up_size,
                    data_format=data_format,
                    interpolation=interpolation)  
    x = upsample(inputs) # 上采样
    return x


# In[18]:


from tensorflow.keras.layers import Input, Flatten, Dense  
from tensorflow.keras.models import Model  
def cnn_model(input_shape):  
    inputs = Input(shape=input_shape)  
    x = conv_layer(inputs, 16, pool_strides=1)  
    x = conv_layer(x, 32, pool_strides=1)  
    x = conv_layer(x, 32, pool_strides=1)  
    x = conv_layer(x, 16)  
    x = upsample_layer(x, 2)  
    x = conv_layer(x, 1, maxpooling=False)  
    y = Flatten(data_format='channels_first')(x)  
    y = Dense(128, activation='relu')(y)  
    y = Dense(128, activation='relu')(y)  
    outputs = Dense(100, activation='relu')(y)  
    # 建立模型  
    model = Model(inputs=inputs, outputs=outputs)  
    return model


# In[19]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error      
from tensorflow.keras.callbacks import EarlyStopping    
SEED = 233  # 随机种子    
np.random.seed(SEED)    
tf.random.set_seed(SEED) 


# In[22]:


data_x = np.load('data_x.npy')    
data_y = np.load('data_y.npy')    
data_x = data_x.astype('float32')    
data_y = data_y.astype('float32')
# data_x = np.moveaxis(data_x, [2, 3], [1, 2])


# In[23]:


# data_x.shape, data_y.shape


# In[24]:


data_x /= 1404 # 归一化  
data_y /= 1404 # 归一化  
train_x, train_y = data_x[:-10800], data_y[:-10800] # 训练集     
test_x, test_y = data_x[-10800:], data_y[-10800:] # 测试集
model = cnn_model(train_x.shape[1:])      
opt = keras.optimizers.Adam(learning_rate=4e-4)      
model.compile(loss='mse',      
              optimizer=opt,      
              metrics=['mae', 'mse'])      
batch_size = 32 # 训练批次大小    
epochs = 50  # 训练轮数    
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # 早停策略    
model.fit(train_x, train_y.reshape(-1, 100),      
        batch_size=batch_size,      
        epochs=epochs,       
        validation_split=0.2,      
        callbacks=[earlystop],      
        shuffle=True) # 模型训练


# In[25]:


predictions_test = model.predict(test_x, batch_size=512)    
mae_test = mean_absolute_error(predictions_test * 1404, test_y.reshape(-1, 100) * 1404)    
mse_test = mean_squared_error(predictions_test * 1404, test_y.reshape(-1, 100) * 1404)    
print(mae_test, mse_test, np.sqrt(mse_test)) # 测试集误差   


# In[42]:


from matplotlib.font_manager import FontProperties
# 中文字体设置，需根据字体的实际路径修改fname参数
fp = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=11)


# In[44]:


fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
for idx, (axi, volframe) in enumerate(zip(ax.ravel(), [
    predictions_test[44].reshape(10,-1),
    test_y[44],
    predictions_test[36].reshape(10,-1),
    test_y[36],
])):
    sns.heatmap(volframe * 1404, vmin=0, cmap='Blues', ax=axi)
    if idx % 2 == 0:
        axi.set_title('预测值', fontproperties=fp)
    else:
        axi.set_title('真实值', fontproperties=fp)
plt.tight_layout()
plt.show()


# In[ ]:




