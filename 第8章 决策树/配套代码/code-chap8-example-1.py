#!/usr/bin/python
#encoding:utf-8

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
dataset=pd.read_csv('example8-1.csv',encoding='gbk')
print(cal_Ent(dataset))


def splitDataSet(data,name, value): 
    reducedFeatVec=data.loc[data[name]==value].drop(name,axis=1)#抽取按name的当前value特征进划分的数据集
    return reducedFeatVec

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
print(chooseBestFeatureToSplit(dataset))