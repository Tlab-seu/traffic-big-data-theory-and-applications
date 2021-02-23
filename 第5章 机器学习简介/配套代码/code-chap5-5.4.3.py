import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
#需要在图中显示中文，需要加入以下两行
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
X = np.random.randint(0, 20, size=[1000,2])  #生成在(0,1)均匀分布的样本值
y = np.random.randint(0, 3, 1000)  #生成只含0,1,2的随机标签
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本量", fontsize=13)
    plt.ylabel("误差值", fontsize=13)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练误差")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证误差")
    plt.legend(loc="best")
    return plt
title = r"学习曲线 (LogisticRegression)"
estimator = LogisticRegression()  
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1)
plt.show() #显示结果如图6-15所示