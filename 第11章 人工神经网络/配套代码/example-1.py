import pandas as pd
df = pd.read_csv("DATASET-B.csv")
# 生成特征
def gerData(df, lag):
    
    jar = []
    for i in range(1, lag+1):
        print(i)
        tmp = df.copy()
        tmp['time_id'] = tmp.time_id.values + i # 每次循环生成t-i时间步的数据
        tmp = tmp.set_index(['rowid', 'colid', 'time_id', 'date'])
        jar.append(tmp)
    #将各时间步数据进行拼接
    jar.append(df[['rowid', 'colid', 'time_id', 'date', 'labels']].set_index(['rowid', 'colid', 'time_id', 'date'])) 
    
    return pd.concat(jar, axis=1).reset_index()

used = ['rowid', 'colid', 'time_id', 'date', 'aveSpeed', 'gridAcc', 'speed_std', 'labels']
dataRaw = df[used] # 筛选标签
data = gerData(dataRaw, 6) # 生成训练数据
data = data.dropna()  # 去除空值


import pandas as pd      
valid_set = data[data.date.between(20161101,20161107)]  #验证集  
train_set = data[data.date.between(20161108,20161121)]  #训练集  
test_set = data[data.date.between(20161122,20161131)]  #测试集  
train_x = train_set.values[:, 4:-1]  #输入  
val_x = valid_set.values[:, 4:-1]  
test_x = test_set.values[:, 4:-1]  
train_y = train_set.values[:, -1]  #输出  
test_y = test_set.values[:, -1]  
val_y = valid_set.values[:, -1] 

from sklearn import preprocessing  
scaler = preprocessing.MinMaxScaler()  
train_x = scaler.fit_transform(train_x)  
val_x = scaler.transform(val_x)  
test_x = scaler.transform(test_x)  

import tensorflow as tf  
from tensorflow.keras.layers import Dense  
model = tf.keras.Sequential()  
model.add(Dense(64,activation = 'relu',input_shape=(24,)))  #30为输入维度  
model.add(Dense(128,activation = 'relu'))  
model.add(Dense(128,activation = 'relu'))  
model.add(Dense(64,activation = 'relu'))      
model.add(Dense(3,activation = 'softmax'))
print(model.summary())  #用于观察模型结构和参数数量  

sgd = tf.keras.optimizers.SGD(learning_rate=0.005)  # 定义优化器
model.compile(loss = 'sparse_categorical_crossentropy',optimizer=sgd,metrics=['accuracy']) # 编译模型   

result = model.fit(train_x,train_y,epochs=20,validation_data = (val_x,val_y),batch_size=1024) # 训练模型
model.evaluate(test_x,test_y)  #评估模型

history = result.history

y_hat = model.predict(test_x)

import numpy as np

label_hat = np.argmax(y_hat, axis=1)

predResult = np.hstack((test_set[['rowid', 'colid']].values, label_hat.reshape(-1, 1), test_y.reshape(-1, 1)))

predResult = predResult.astype(int)

rows = predResult[:, 0].max() + 1
cols = predResult[:, 1].max() + 1
matPred = np.ones((rows, cols)) + 2
matTrue = np.ones((rows, cols)) + 2

for i in range(len(predResult)):
    row = predResult[i][0]
    col = predResult[i][1]
    matPred[row, col] = predResult[i][2]
    matTrue[row, col] = predResult[i][3]

matPred = matPred.astype(int)
matTrue = matTrue.astype(int)


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=2)
cmap = cmap = sns.cm.rocket
ticks = range(4)
plt.figure(figsize=(9, 6))
sns.heatmap(matPred, cmap=cmap, cbar_kws={"ticks":ticks}, xticklabels=False, yticklabels=False)
plt.figure(figsize=(9, 6))
sns.heatmap(matTrue, cmap=cmap, cbar_kws={"ticks":ticks}, xticklabels=False, yticklabels=False)

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pylab import mpl
sns.set(style='white')
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体


train_loss = history['loss']
train_acc = history['acc']
val_loss = history['val_loss']
val_acc = history['val_acc']

plt.figure(figsize=(7, 5))
plt.plot(range(len(train_loss)), train_loss, linewidth=2.5)
plt.plot(range(len(val_loss)), val_loss, linewidth=2.5)
plt.legend(['训练集误差', '验证集误差'], fontsize=16)
plt.xlabel('迭代次数', fontsize=16)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(len(train_acc)), train_acc, linewidth=2.5)
plt.plot(range(len(val_acc)), val_acc, linewidth=2.5)
plt.legend(['训练集准确率', '验证集准确率'], loc='upper right', fontsize=16)
plt.xlabel('迭代次数', fontsize=16)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.show()