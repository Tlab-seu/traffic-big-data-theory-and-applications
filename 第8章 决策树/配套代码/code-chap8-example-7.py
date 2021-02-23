import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'  #注意修改为安装Graphviz的路径
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus
data=pd.read_csv('example8-2.csv',encoding='gbk')
x1=data['平均速度']
x2=data['流量']
x3=pd.Categorical(data['是否停车']).codes
x=np.stack((x1,x2),axis=1)
y=pd.Categorical(data['交通状态']).codes
model = DecisionTreeClassifier(criterion='gini')
model.fit(x, y)
traffic_feature_E = 'speed_class','volume_class'
label = 'congestion', 'smooth','amble'
with open('traffic_condition.dot','w') as f: 
    tree.export_graphviz(model, out_file=f)
    dot_data = tree.export_graphviz(model, out_file=None, 
                                    feature_names=traffic_feature_E, 
                                    class_names=label,
                                    filled=True, rounded=True, 
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    f = open('traffic_condition_decision_tree.png', 'wb')
    f.write(graph.create_png())
    f.close()

