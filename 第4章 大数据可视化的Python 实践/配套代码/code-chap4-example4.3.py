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
