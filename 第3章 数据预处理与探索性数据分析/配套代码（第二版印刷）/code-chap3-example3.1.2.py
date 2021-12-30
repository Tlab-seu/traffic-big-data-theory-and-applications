import pandas as pd   
import numpy as np   
# 拉格朗日插值        
from scipy.interpolate import lagrange  #scipy.interpolate是内置工具包        
def ploy (s,n,k=5):        
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))] #取出插值位置前后k个数据  
    y=y[y.notnull()]  #剔除空值      
    return lagrange(y.index,list(y))(n)  
traj = pd.read_csv('DATASET-A.csv', header=None, usecols=[2,3,4]).iloc[:15]  
traj.columns = ['timestamp', 'lon', 'lat']  
traj['time_interval'] = traj['timestamp'] - traj['timestamp'].shift(1)    
index = traj[traj['time_interval'] >=6].index.to_list()    
for i in index:    
    timestamp = traj['timestamp'].loc[i-1] + 3    
    insertRow = pd.DataFrame([[np.nan, np.nan, timestamp]], columns=['lon', 'lat', 'timestamp'])    
    traj = pd.concat([traj[:i], insertRow, traj[i:]], ignore_index=True)    
    traj['lon'][i]=ploy(traj['lon'],i)    
    traj['lat'][i]=ploy(traj['lat'],i)    
traj = traj.drop(['time_interval'], axis=1) 