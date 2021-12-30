import pandas as pd 
data = pd.read_csv('DATASET-B.csv')
data_speed = data[(data['date']==20161101) & (data['time_id']==0)]['aveSpeed']
statistics = data_speed.describe()#保存基本统计量      
statistics.loc['range']=statistics.loc['max']-statistics.loc['min']#极差 
statistics.loc['var']=statistics.loc['std']/statistics.loc['mean']#变异系数  
statistics.loc['dis']=statistics.loc['75%']-statistics.loc['25%']#四分位数间距 
print (statistics)
statistics.to_csv('statistics.csv')