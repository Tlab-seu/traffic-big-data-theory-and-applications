import pandas as pd
from math import log
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
#连续特征处理
def splitDataSet_c(dataSet, name, value, LorR='L'):
    retDataSet = []
    featVec = []
    if LorR == 'L':
        retDataSet=dataSet.loc[dataSet[name]<value].drop(name,axis=1)
    else:
        retDataSet=dataSet.loc[dataSet[name]>value].drop(name,axis=1)
    return retDataSet


def chooseBestFeatureToSplit_c(dataSet):
    baseEntropy = cal_Ent(dataSet)  # 计算根节点的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    for name in dataset.columns[:-2]:#取平均速度和流量两个特征
        uniqueVals=set(dataset[name])# 获取当前特征的所有可能取值
        newEntropy = 0.0
        bestPartValuei = None
        sortedUniqueVals = list(uniqueVals)  # 对特征值排序
        sortedUniqueVals.sort()
        listPartition = []
        minEntropy = float("inf")
        print(len(sortedUniqueVals))
        for j in range(len(sortedUniqueVals) - 1): 
            partValue = (float(sortedUniqueVals[j]) + float(sortedUniqueVals[j + 1]))/2 #计算划分点
            dataSetLeft = splitDataSet_c(dataSet, name, partValue, 'L')
            dataSetRight = splitDataSet_c(dataSet, name, partValue, 'R')
            probLeft = len(dataSetLeft) / float(len(dataSet))
            probRight = len(dataSetRight) / float(len(dataSet))
            Entropy = probLeft * cal_Ent(dataSetLeft) + probRight * cal_Ent(dataSetRight)#计算信息熵
            print(name,partValue,baseEntropy,cal_Ent(dataSetLeft),cal_Ent(dataSetRight),baseEntropy - Entropy)
#             print(name,partValue,baseEntropy - Entropy)

            if Entropy < minEntropy:  # 取最小的信息熵
                minEntropy = Entropy
                bestPartValuei = partValue
            newEntropy = minEntropy
            infoGain = baseEntropy - newEntropy  # 计算信息增益
        if infoGain > bestInfoGain:  # 取最大的信息增益对应的特征
            bestInfoGain = infoGain
            bestFeature = name
            bestPartValue = bestPartValuei
    return bestFeature, bestPartValue

dataset=pd.read_csv('example8-2.csv',encoding='gbk')
print(chooseBestFeatureToSplit_c(dataset))