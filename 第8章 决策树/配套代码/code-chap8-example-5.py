import pandas as pd
from math import log
import pandas as pd
from operator import itemgetter
def cal_Ent(data):
    sample_size = len(data)  # 样本数
    labelCounts = {}   # 创建字典，key是交通状态类别，value是属于该类别的样本个数
    for index,data in data.iterrows(): # 遍历整个数据集，每次取一行
        currentLabel = data['交通状态']  #取标签的值
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0  # 初始化信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/sample_size
        Ent -= prob * log(prob,2) #计算信息熵
    return Ent
def splitDataSet(data,name, value): 
    reducedFeatVec=data.loc[data[name]==value].drop(name,axis=1)#抽取按name的当前value特征进划分的数据集
    return reducedFeatVec
def majorityCnt(classList):
    classCount={}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)  
    return sortedClassCount[0][0]  
def chooseBestFeatureToSplit(dataset):
    baseEntropy = cal_Ent(dataset)  #计算当前数据集的信息熵
    bestInfoGain = 0.0 #初始化最优信息增益和最优的特征
    bestFeature = -1
    for name in dataset.columns[:-1]:
        uniquevals=set(dataset[name])# 获取当前特征的所有可能取值
        newEntropy = 0.0
        for value in uniquevals:#计算每种划分方式的信息熵
            subDataSet=splitDataSet(dataset,name,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * cal_Ent(subDataSet)
        infoGain = baseEntropy - newEntropy #计算信息增益
        print(name,newEntropy)
        if (infoGain >=bestInfoGain):     #比较每个特征的信息增益，只要最好的信息增益
            bestInfoGain = infoGain
            bestFeature = name
    return bestFeature

def createTree(dataSet,featureName):
    classList = dataSet['交通状态'].tolist()
    if classList.count(classList[0]) == len(classList): # 统计属于列别classList[0]的个数
        return classList[0] # 当类别完全相同则停止继续划分
    if len(dataSet.iloc[0]) ==1: # 当只有一个特征的时候，遍历所有实例返回出现次数最多的类别
        return majorityCnt(classList) # 返回类别标签
    print(dataSet)
    bestFeatLabel= chooseBestFeatureToSplit(dataSet)#最佳特征
    print(bestFeatLabel)
    myTree ={bestFeatLabel:{}}  # map 结构，且key为featureLabel
    featureName.remove(bestFeatLabel)
    # 找到需要分类的特征子集
    featValues =dataSet[bestFeatLabel]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = featureName[:] # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeatLabel,value),subLabels)
    return myTree

featureName=['平均速度','流量','是否停车']
dataset=pd.read_csv('example8-1.csv',encoding='gbk')
mytree=createTree(dataset,featureName)
print(mytree)

