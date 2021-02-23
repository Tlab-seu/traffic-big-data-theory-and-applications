import numpy as np
from sklearn.metrics import precision_recall_curve as prc
def precision_recall():
    y_true = np.random.randint(0, 2, 50)  #随机生成50个样本的标签，取值0或1
    y_scores = np.random.uniform(0, 1, 50) #随机生成每个样本的置信度
    precision, recall, thresholds = prc(y_true, y_scores) #调用precision_recall_curve
    return y_true, y_scores, precision, recall, thresholds
 
model=['模型A','模型B','模型C']
y_true1, y_scores1, precision1, recall1, thresholds1=precision_recall() #模型A
y_true2, y_scores2, precision2, recall2, thresholds2=precision_recall() #模型B
y_true3, y_scores3, precision3, recall3, thresholds3=precision_recall() #模型C
import matplotlib.pyplot as plt  #下面画出三个模型的P-R曲线
#需要在图中显示中文，需要加入以下两行
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('召回率', fontsize =14)
plt.ylabel('精确率', fontsize =14)
plt.plot(recall1,precision1,recall2,precision2,recall3,precision3)
plt.xlim([0, 1.0])
plt.ylim([0, 1.05])
plt.legend(model, loc="lower right",fontsize=12 )
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') #斜率为1的虚线，即平衡点
plt.savefig('p-r.png')
plt.show()  #显示结果如图6-9（a）所示
 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_scores1)#模型A
fpr2, tpr2, thresholds2 = roc_curve(y_true2, y_scores2)#模型B
fpr3, tpr3, thresholds3 = roc_curve(y_true3, y_scores3)#模型C
def plot_roc_curve(fpr, tpr, model):
    roc_auc = auc(fpr, tpr) #计算模型的AUC面积
    plt.plot(fpr, tpr, label='%s (area= %0.2f)' %(model, roc_auc))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('假正类率（FPR）', fontsize =14)
    plt.ylabel('真正类率（TPR）', fontsize =14)
plot_roc_curve(fpr1, tpr1, model[0])#模型A
plot_roc_curve(fpr2, tpr2, model[1])#模型B
plot_roc_curve(fpr3, tpr3, model[2])#模型C
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right", fontsize=12)
plt.savefig('ROC.png')
plt.show() #显示结果如图6-9（b）所示
