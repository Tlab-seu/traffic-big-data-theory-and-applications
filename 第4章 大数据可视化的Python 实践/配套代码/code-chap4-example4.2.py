#%%  
#线型图  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.sans-serif'] = 'SimHei' 
data_all = pd.read_csv('DATASET-B.csv')  
data_20161101_02 = data_all[(data_all['date']==20161101) | (data_all['date']==20161102)]  
fig = plt.figure(figsize=(10,6))  
x = np.arange(1, 145)  
y1 = data_20161101_02[data_20161101_02['date']==20161101].groupby('time_id')['aveSpeed'].mean()  
y2 = data_20161101_02[data_20161101_02['date']==20161102].groupby('time_id')['aveSpeed'].mean()  
plt.plot(x, y1, label='20161101')  
plt.plot(x, y2, label='20161102')  
plt.xticks(np.arange(0,145,20))  
plt.legend(fontsize=18, loc='upper right')    
plt.tick_params(labelsize=18)    
plt.xlabel('时间', fontsize=20)    
plt.ylabel('平均速度', fontsize=20)  
plt.savefig('线型图') 

#%%
#条形图1%%  
data = data_20161101_02[(data_20161101_02['date']==20161101) & (data_20161101_02['time_id']==50)]  
data = data.reset_index(drop=True)  
data['stop_or_not'] = data['stopNum'].apply(lambda x: 0 if x==0 else 1)  
data['volume_CATE'] = data['volume']//5
fig = plt.figure(figsize=(10,6))  
sns.barplot(x='volume_CATE', y='aveSpeed', data=data, color='#1f77b4')  
x_bar = np.arange(10)  
y = data_20161101_02[data_20161101_02['date']==20161101].groupby('time_id')['aveSpeed'].mean()  
volume = ['0~5','5~10','10~15','15~20','20~25','25~30','30~35','35~40','40~45','45~50']  
plt.xticks(x_bar, volume, rotation=45)  
plt.ylim(bottom=3)  
plt.tick_params(labelsize=16)   
plt.xlabel('流量', fontsize=20)  
plt.ylabel('平均速度', fontsize=20) 
fig.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9) 
fig.savefig('条形图1.png')  


#%%    
#条形图2  
#sns.set(style='whitegrid')    

fig = plt.figure(figsize=(10,6))  
sns.barplot(x='volume_CATE', y='aveSpeed', data=data, hue="stop_or_not")  
plt.ylim(bottom=3)  
plt.tick_params(labelsize=18)   
plt.xlabel('流量', fontsize=20)  
plt.ylabel('平均速度', fontsize=20)  
fig.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
fig.savefig('条形图2.png') 

#%%  
#条形图3     
fig = plt.figure(figsize=(10,6))    
sns.countplot(x='volume_CATE', data=data, color='#1f77b4')  
x_bar = np.arange(10)    
y = data_20161101_02[data_20161101_02['date']==20161101].groupby('time_id')['aveSpeed'].mean()    
volume = ['0~5','5~10','10~15','15~20','20~25','25~30','30~35','35~40','40~45','45~50']  
plt.xticks(x_bar, volume, rotation=45)  
plt.tick_params(labelsize=16)   
plt.xlabel('流量', fontsize=20)  
plt.ylabel('频数', fontsize=20)  
fig.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
fig.savefig('条形图3.png')

#%%  
#箱型图  
#sns.set(style='whitegrid')  
x_bar = np.arange(10)  
fig = plt.figure(figsize=(10,6))  
sns.boxplot(x='volume_CATE', y='aveSpeed', data=data, color='#1f77b4')  
plt.xticks(x_bar, volume, rotation=45)  
plt.tick_params(labelsize=16)   
plt.xlabel('流量', fontsize=20)  
plt.ylabel('平均车速', fontsize=20)  
fig.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
fig.savefig('箱型图.png')  

#%%  
#饼图  
#sns.set(style='whitegrid')  

# data['volume_CATE2'] = data['volume_CATE'].apply(lambda x: x if x<10 else -1)
count = data.groupby(by='volume_CATE').size()  
fig = plt.figure(figsize=(6,6))
plt.pie(x=list(count),
        labels=['0~5','5~10','10~15','15~20','20~25','25~30','30~35','35~40','40~45','45~50'],
        explode=[0,0,0,0,0,0,0,0,0,0.2], 
        colors=sns.color_palette('Blues', n_colors=11),   
        labeldistance=1.2, autopct='%.2f%%',   
        pctdistance=0.8, radius=2,
        textprops={'fontsize':20, 'weight':'bold'},shadow=True)  
plt.savefig('饼图.png',dpi=200, bbox_inches='tight')

#%%  
#直方图1    
fig = plt.figure(figsize=(10,6))  
sns.distplot(data['aveSpeed'])  
plt.tick_params(labelsize=18)  
plt.xlabel('平均速度', fontsize=20)  
plt.ylabel('概率', fontsize=20)  
fig.savefig('直方图1.png')  

#%%      
#直方图2    
#sns.set(style='whitegrid')      
fig = plt.figure(figsize=(10,6))  
sns.distplot(data[data['stop_or_not']==0]['aveSpeed'], label='停车次数为0')  
sns.distplot(data[data['stop_or_not']==1]['aveSpeed'], label='停车次数不为0')  
plt.legend(fontsize=15, loc='upper right')  
plt.tick_params(labelsize=18)  
plt.xlabel('平均速度', fontsize=20)  
plt.ylabel('概率', fontsize=20)  
fig.savefig('直方图2.png') 

#%%  
#散点图1   
fig = plt.figure(figsize=(10,6))  
plt.scatter(data['aveSpeed'], data['volume'], alpha=0.6)
plt.tick_params(labelsize=18)  
plt.xlabel('平均速度', fontsize=20)  
plt.ylabel('流量', fontsize=20)  
fig.savefig('散点图1.png') 

#%%  
#散点图2  
#sns.set(style='darkgrid')  
# fig = plt.figure(figsize=(10,6))
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig = sns.jointplot(x='aveSpeed', y='volume', data=data, alpha=0.6)
fig.set_axis_labels('平均速度','流量') 
fig.savefig('散点图2.png')   

#%%  
#散点图3   
sns.axes_style("white")  
fig = sns.jointplot(x='aveSpeed', y='volume',
                        data=data, kind='hex')  
fig.set_axis_labels('平均速度','流量')  
fig.savefig('散点图3.png')   

#%%  
#散点图4 
data2 = data.copy(deep=True)
data2.rename(columns={'volume': '流量', 'aveSpeed': '平均速度', 'gridAcc': '平均加速度'}, inplace=True)  
sns.set_style('white',{'font.sans-serif':['simhei']})  
fig = sns.pairplot(data2[['流量','平均速度','平均加速度']])
fig.savefig('散点图4.png', dpi=300)  

#%%
tips = sns.load_dataset("tips")    
fig = plt.figure(figsize=(10,6))    
sns.regplot(x="total_bill", y="tip", data=tips, color='#1f77b4')    
plt.xlabel('账单', fontsize=20)    
plt.ylabel('小费', fontsize=20)
fig.savefig('reg.png') 

#%%
#分面图形
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig = sns.catplot('volume_CATE', col='stop_or_not', data=data,       
                  kind='count', color='#1f77b4')      
fig.set_xticklabels(volume, rotation=45)      
fig.set_titles('是否停车: {col_name}')      
fig.set_axis_labels(x_var='流量', y_var='数量')
fig.savefig('分面图形.png')    

#%%
fig = plt.figure(figsize=(10,10))    
ax1 = plt.subplot(2,2,1)    
ax2 = plt.subplot(2,2,2)    
ax3 = plt.subplot(2,1,2)    
f1 = sns.countplot(x='volume_CATE', data=data[data['stop_or_not']==0],     
                   color='#1f77b4', ax=ax1)    
f1.set_xticklabels(volume, rotation=45, fontsize=15)    
f1.set_title('停车次数为0', fontsize=20)    
f1.set_xlabel('流量', fontsize=20)    
f1.set_ylabel('数量', fontsize=20)    
f2 = sns.countplot(x='volume_CATE', data=data[data['stop_or_not']==1],     
                   color='#1f77b4', ax=ax2)    
f2.set_xticklabels(volume, rotation=45, fontsize=13)    
f2.set_title('停车次数不为0', fontsize=20)    
f2.set_xlabel('流量', fontsize=20)    
f2.set_ylabel('数量', fontsize=20)    
ax3.scatter(data['aveSpeed'], data['volume'], alpha=0.2)    
ax3.set_xticklabels(x, fontsize=18)    
ax3.set_xlabel('平均速度', fontsize=20)    
ax3.set_ylabel('流量', fontsize=20)    
ax3.set_title('平均速度-流量图', fontsize=20)    
fig.tight_layout(w_pad=0.5, h_pad=2)    
fig.savefig('分面图形2.png')

#%%  
#三维图形    
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,8))  
ax = fig.gca(projection='3d') 
data_1 = data[data['stop_or_not']==0]  
data_0 = data[data['stop_or_not']==1]  
ax.scatter(data_1['volume'], data_1['aveSpeed'], data_1['gridAcc'], label='1')  
ax.scatter(data_0['volume'], data_0['aveSpeed'], data_0['gridAcc'], label='0')  
ax.set_xlabel('流量', labelpad=15, fontsize=20)  
ax.set_ylabel('平均速度', labelpad=15, fontsize=20)  
ax.set_zlabel('平均加速度', labelpad=15, fontsize=20)  
ax.legend(['停车次数为0', '停车次数不为0'], loc='upper right', fontsize=20)  
fig.savefig('3d.png', dpi=300)  

#%%
fig = plt.figure(figsize=(10,8))    
ax = fig.gca(projection='3d')    
X = np.arange(-2, 2, 0.1)    
Y = np.arange(-2, 2, 0.1)    
X, Y = np.meshgrid(X, Y)    
Z = np.sqrt(X**2 + Y**2)    
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)    
ax.set_xlabel('X', labelpad=15, fontsize=20)    
ax.set_ylabel('Y', labelpad=15, fontsize=20)    
ax.set_zlabel('Z', labelpad=15, fontsize=20) 
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.zaxis.set_tick_params(labelsize=16)
fig.savefig('3d2.png')

#%%
from bokeh.plotting import figure  
from bokeh.io import output_notebook, show
import numpy as np
import pandas as pd 
data_all = pd.read_csv('DATASET-B.csv')  
data_20161101_02 = data_all[(data_all['date']==20161101) | (data_all['date']==20161102)]    
data = data_20161101_02[(data_20161101_02['date']==20161101) & (data_20161101_02['time_id']==50)]    
data = data.reset_index(drop=True)
data['stop_or_not'] = data['stopNum'].apply(lambda x: 0 if x==0 else 1)   
data['volume_CATE'] = data['volume']//5 
grouped = data.groupby('volume_CATE')  
speed = grouped.aveSpeed
stop = list(grouped.groups) 
avg, std = speed.mean(), speed.std()  
stop_0 = data[data['stop_or_not']==0]  
stop_1 = data[data['stop_or_not']==1]  
p = figure(x_axis_label='流量', y_axis_label='速度')  
p.vbar(x=stop, bottom=avg-std, top=avg+std, width=0.8,   
       fill_alpha=0.2, line_color=None, legend="平均速度 标准差")  
p.circle(x=stop_0["volume_CATE"], y=stop_0["aveSpeed"], size=5, alpha=0.8,  
        color=(73,132,175), legend="停车次数为0")  
p.square(x=stop_1["volume_CATE"], y=stop_1["aveSpeed"], size=5, alpha=0.8,  
        color=(255,132,23), legend="停车次数不为0")  
p.legend.location = "top_left"  
show(p)


# %%
from bokeh.models import ColumnDataSource  
from bokeh.layouts import gridplot    
source = ColumnDataSource(data)  
options = dict(plot_width=300, plot_height=300,  
               tools="pan,wheel_zoom,box_zoom,box_select,lasso_select")  
p1 = figure(title="流量 vs. 平均速度", **options,  
            x_axis_label='流量', y_axis_label='平均速度')  
p1.circle("volume", "aveSpeed", color=(73,132,175), source=source)  
p2 = figure(title="流量 vs. 平均加速度", **options,  
            x_axis_label='流量', y_axis_label='平均加速度')  
p2.circle("volume", "gridAcc", color="cadetblue", source=source)   
p3 = figure(title="平均速度 vs. 平均加速度", **options,  
            x_axis_label='平均速度', y_axis_label='平均加速度')  
p3.circle("aveSpeed", "gridAcc", color=(255,132,23), fill_color=None, source=source)   
p = gridplot([[ p1, p2, p3]], toolbar_location="right")  
show(p)  


# %%
y1 = data_20161101_02[data_20161101_02['date']==20161101].groupby('time_id')['aveSpeed'].mean()  
y2 = data_20161101_02[data_20161101_02['date']==20161102].groupby('time_id')['aveSpeed'].mean()  
x = np.arange(1,145)
p = figure(title='20161101和20161102平均速度时变图', plot_width=500, plot_height=300, x_axis_label='时间', y_axis_label='平均速度')  
p.line(x, y1, legend="2016/11/01", color=(73,132,175), line_width=3)  
p.line(x, y2, legend="2016/11/02", color=(255,132,23), line_width=3)  
p.legend.location = "top_right"  
show(p)  


# %%
