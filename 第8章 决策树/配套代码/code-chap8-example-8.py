import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn import preprocessing



# 数据文件路径
data = pd.read_csv('DATASET-B.csv',nrows=200000)
x = data[['aveSpeed','stopNum','volume','speed_std']]
y = pd.Categorical(data['labels']).codes
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,random_state=1)

# 决策树参数估计
model = DecisionTreeClassifier(criterion='gini', min_samples_split=10,
                               min_samples_leaf=40,max_depth=10,
                               class_weight='balanced')
model.fit(x_train, y_train)

print(model.feature_importances_)

# 测试集上的预测结果
y_test_hat = model.predict(x_test)# 测试数据
y_test = y_test.reshape(-1)
result = (y_test_hat == y_test)# True则预测正确，False则预测错误
acc = np.mean(result)
print('准确度: %.2f%%' % (100 * acc))


#画图
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
N, M = 50, 50  # 横纵各采样多少个值
x1_min, x2_min,_ ,_  = x.min()
x1_max, x2_max,_,_ = x.max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
model2 = DecisionTreeClassifier(criterion='gini', min_samples_split=10,
                               min_samples_leaf=40,max_depth=10,
                               class_weight='balanced')
model2.fit(x_train[['aveSpeed','stopNum']],y_train)
y_show_hat = model2.predict(x_show)  # 预测值
y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
# 随机选取 2000个样本点  
sample_plot_idx = np.random.choice(x_test.shape[0],size=2000,replace=False) 
x_test1=x_test.iloc[sample_plot_idx]
y_test1=y_test[sample_plot_idx]

plt.figure(facecolor='w',dpi=300)
plt.pcolormesh(x1, x2, y_show_hat, alpha=0.1)  # 预测值的显示
condition=['畅通','缓行','拥堵']
color=['purple','green','yellow']
for index in range(3):
    plot_idx=np.where(y_test1==index)
    plt.scatter(x_test1.iloc[plot_idx]['aveSpeed'],x_test1.iloc[plot_idx]['stopNum'],c=color[index],edgecolors='k',s=20, zorder=10,label=condition[index]) 

plt.xlabel(u'速度(m/s)', fontsize=12)
plt.ylabel( u'平均停车次数', fontsize=12)
plt.xlim(x1_min-0.3, x1_max+0.3)
plt.ylim(x2_min-0.1, x2_max+0.1)
plt.title(u'道路交通状态的决策树分类', fontsize=12)
plt.legend()
plt.savefig('result.jpg')
plt.show()