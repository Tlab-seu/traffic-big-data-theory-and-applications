# 2.8.4	Pandas简介
import pandas as pd    
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df.columns = ['A', 'B', 'C']  #定义表头
df['A']
df[['A', 'B']]
df[1:3]         #取1到2行，输出为

df.iloc[1:3, 2]   #多维索引，输出为
oc    
df[df['A']>1]   #根据条件选择，输出为

print(type(df.A))     # 输出<class 'pandas.core.series.Series'>
tmp = pd.Series([1, 2, 3], name='tmp')  # 创建一个名为tmp的Series
# 文件的读取与写入
df = pd.read_csv('file.csv')    				#读取csv格式文件，自动识别第一行为表头  
df = pd.read_csv('file.csv', names=['A','B','C']) 	#指定表头  
df = pd.read_csv('file.csv', index_col=0)    		#指定索引列  
df.to_csv('newfile.csv')            			#在工作目录下新建csv文件  
# 对列进行操作
#（1）创建新列
df['new'] = 1               				#新建一列（添加为最后一列），赋值为1
df['new1'] = range(len(df)) 					#新建一列，赋值为由0开始的递增序列
df['date'] = ['20190801', '20190702', '20190601'] 	#新建一列存储日期
#（2）基于已有列进行计算
df['new'] = df['A'] + 1               		#新建一列，赋值为列A加1
df['new'] = df['A'] + df['B']        		#新建一列，赋值为列A与列B的和
#新建一列，赋值为列A与列B的商，转换为numpy数组进行计算，能提升性能
df['new'] = df['A'].values / df['B'].values 
# 在其他计算较为复杂的情况下，需要利用apply语句，传入自定义函数进行列的运算，通常来说，此种用法更为通用。
#任务：新建一列，将date字段的月份取出，并转换为int类型，此处使用了匿名函数，参见3.7.1，也可直接传入已有函数的函数名
df['new'] = df['date'].apply(lambda x: int(x[4:6]))
#任务：新建一列，对两列进行apply操作，获得列A与列B的和
df['new'] = df[['A','B']].apply(lambda x: x[0]+x[1], axis=1)
#	分组操作
#划分  
grouped = df.groupby(by=[ 'A'])  	#按列A进行分组  
grouped['C']            		#取分组后的C列  
df['C'].groupby(df['A'])       	#等价于上面两行语句  
for name, group in grouped:  
    print(name)  
    print(group)  
    grouped.get_group('xx')     #直接获取列A的值为’xx’的组，例如：grouped.get_group(1)
grouped.sum()    	#对各组进行单独求和
grouped.size()   		#获取各组的大小
grouped.mean()   	#获取各组的均值
grouped.min()    	#获取各组最小值
grouped.max()       	#获取各组最大值
grouped.std()       	#获取各组标准差  
grouped.var()       	#获取各组方差  
#使用自定义函数  
grouped.aggregate(f)  	#f也可采用匿名函数的形式  


df = pd.DataFrame({'VehicleType': ['Car', 'Truck', 'Car', 'Truck'], 'Speed': [67., 43., 72., 49.]})
df.groupby(['VehicleType']).mean()

df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)

df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
