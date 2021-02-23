from utm import *    
from tqdm import tqdm, tqdm_pandas
import pandas as pd
import numpy as np
import time 
time1 = '20161101 08:00:00'    
time2 = '20161101 09:00:00'
stamp1 = time.mktime(time.strptime(time1, "%Y%m%d %H:%M:%S")) 
stamp2 = time.mktime(time.strptime(time2, "%Y%m%d %H:%M:%S"))
df = pd.read_csv('DATASET-A.csv', header=None)  
df.columns=['driver_id', 'order_id', 'timestamp', 'lon', 'lat']
# 转换为utc+8时区
df.timestamp = df.timestamp + 8 * 3600
df = df[(df['timestamp'] >= stamp1) & (df['timestamp'] < stamp2)].reset_index(drop=True)
print (df.info())
print (df.head(10))

from osgeo import osr
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)  #wgs-84坐标系
inp = osr.SpatialReference()
inp.ImportFromEPSG(3857)    #Pseudo-Mercator坐标系
# 定义坐标转换
transformation = osr.CoordinateTransformation(wgs84, inp)
#转换坐标
xy = df[['lon', 'lat']].apply(lambda x: transformation.TransformPoint(x[0], x[1])[:2], axis=1) 
# xy为一个list，每一个元素为一个tuple
# 转换为dataframe中的两列
df['x'] = [x[0] for x in xy]
df['y'] = [x[1] for x in xy]

#时间窗划分 
time_interval=600 #时间窗长度  
df['time_id'] = df['timestamp'].apply(lambda x: (x - stamp1)//time_interval) #生成时间窗索引  
#空间网格划分    
left = df['x'].min() #计算左边界    
up = df['y'].max() #计算上边界    
interval=70 #网格单元大小    
df['rowid'] = df['y'].apply(lambda x: (up - x) // interval).astype('int') #计算横向索引    
df['colid'] = df['x'].apply(lambda x: (x - left) // interval).astype('int')#计算纵向索引 

df = df.sort_values(by=['driver_id', 'order_id', 'timestamp']).reset_index(drop=True)  
# 将订单id，下移一行，用于判断相邻记录是否属于同一订单  
df['orderFlag'] = df['order_id'].shift(1)  
df['identi'] = (df['orderFlag']==df['order_id'])  
# 将坐标、时间戳下移一行，从而匹配相邻轨迹点  
df['x1'] = df['x'].shift(1)  
df['y1'] = df['y'].shift(1)  
df['timestamp1'] = df['timestamp'].shift(1)
df = df[df['identi']==True]   #将不属于同一订单的轨迹点对删去  
dist = np.sqrt(np.square((df['x'].values-df['x1'].values)) + np.square((df['y'].values-df['y1'].values)))    # 计算相邻轨迹点之间的距离    
time = df['timestamp'].values - df['timestamp1'].values   # 计算相邻轨迹点相差时间  
df['speed'] = dist / time    # 计算速度    
df = df.drop(columns=['x1', 'y1', 'orderFlag', 'timestamp1', 'identi'])   # 删去无用列 

df['speed1'] = df.speed.shift(1)                 # 将速度下移一行
df['timestamp1'] = df.timestamp.shift(1)         # 将时间下移一行
df['identi'] = df.order_id.shift(1)              # 将订单号下移一行
df = df[df.order_id==df.identi]                  # 去除两个订单分界点数据
df.loc[:, 'acc'] = (df.speed1.values - df.speed.values) / (df.timestamp1.values - df.timestamp.values)  #计算加速度
df = df.drop(columns=['speed1', 'timestamp1', 'identi'])  #删除临时字段

orderGrouped = df.groupby(['rowid', 'colid','time_id', 'order_id'])  # 基于时空网格与轨迹id进行分组   
# 网格平均车速  
grouped_speed = orderGrouped.speed.mean().reset_index()  
grouped_speed = grouped_speed.groupby(['rowid', 'colid', 'time_id'])  
grid_speed = grouped_speed.speed.mean()  
grid_speed = grid_speed.clip(grid_speed.quantile(0.05), grid_speed.quantile(0.95))#去除异常值  
grid_speed.head()  

# 网格平均加速度
gridGrouped = df.groupby(['rowid', 'colid','time_id'])
grid_acc = gridGrouped.acc.mean()  
grid_acc.head() 

# 网格流量  
grouped_volume = orderGrouped.speed.last().reset_index()  
grouped_volume = grouped_volume.groupby(['rowid', 'colid', 'time_id'])  
grid_volume = grouped_volume['speed'].size()  
grid_volume = grid_volume.clip(grid_volume.quantile(0.05), grid_volume.quantile(0.95))  
grid_volume.head() 

# 网格车速标准差  
grid_v_std = gridGrouped.speed.std()  
grid_v_std.head()  

# 网格平均停车次数  
stopNum = gridGrouped.speed.agg(lambda x: (x==0).sum())  
grid_stop = pd.concat((stopNum, grid_volume), axis=1)  
grid_stop['stopNum'] = stopNum.values / grid_volume.values  
grid_stop = grid_stop['stopNum']  
grid_stop = grid_stop.clip(0,grid_stop.quantile(0.95))  
grid_stop.head()  

feature = pd.concat([grid_speed, grid_acc, grid_volume, grid_v_std, grid_stop], axis=1).reset_index()  
feature.columns = ['rowid', 'colid', 'time_id', 'aveSpeed', 'gridAcc', 'volume', 'speed_std', 'stopNum']  
feature.head()  