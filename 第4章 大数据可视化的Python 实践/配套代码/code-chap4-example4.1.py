#%%
import matplotlib.pyplot as plt 

#%%
# 分面图形引入
plt.axis([-1, 1, -10, 10])
ax1 = plt.axes([0.55, 0.6, 0.25, 0.2], facecolor='cadetblue')
ax2 = plt.axes([0.2, 0.6, 0.25, 0.2], facecolor='cornflowerblue')
ax3 = plt.axes([0.55, 0.2, 0.25, 0.2], facecolor='cadetblue')
ax4 = plt.axes([0.2, 0.2, 0.25, 0.2], facecolor='cornflowerblue')

#%%
ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

#%%
plt.axis([-1, 1, -10, 10])
plt.grid(b=None, which='major', axis='y')
plt.axhspan(0,10)

#%%
x = range(100)
y = [i**2 for i in x]
plt.plot(x,y)

#%%
x = range(10)
y = [i**2 for i in x]
plt.plot(x, y, marker='o', markersize=5, 
        markerfacecolor='darkred', markeredgewidth=0, 
        ls='-.', c='cornflowerblue')

#%%
# 坐标轴设置
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ax = plt.subplot()
xmajorLocator = MultipleLocator(2.5)
xmajorFormatter = FormatStrFormatter('%.1f')
ymajorLocator = MultipleLocator(20)
ymajorFormatter = FormatStrFormatter('%.1f')

x = range(10)
y = [i**2 for i in x]
plt.plot(x,y)

ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.set_xlim(left=0, right=10)
ax.set_ylim(bottom=0)

#%%
# 坐标轴设置2
x = range(10)
y = [i**2 for i in x]
plt.plot(x,y)
plt.xticks([0, 2.5, 5, 7.5])  
plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0]) 
plt.xlim(left=0, right=10)
plt.ylim(bottom=0)


#%%

plt.plot(x, y, marker='o', markersize=5, 
        markerfacecolor='darkred', markeredgewidth=0, 
        ls='-.', c='cornflowerblue')

ax.xaxis.grid(True)
ax.yaxis.grid(True)
#%%
x = range(10)
y = [i**2 for i in x]
plt.plot(x, y, marker='o', markersize=5, 
        markerfacecolor='darkred', markeredgewidth=0, 
        ls='-.', c='cornflowerblue')
plt.xticks([0, 2.5, 5, 7.5])
plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0])
plt.grid()

#%%
# 添加图例和注释
import numpy as np
np.random.seed(123)
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 2, 100)
x3 = np.random.normal(10, 1, 100)

plt.plot(x1, label = '1st')
plt.plot(x2, label = '2nd')
plt.plot(x3, label = '3rd')

plt.legend(loc='upper right')
plt.annotate('important value', (53, 18), xytext=(40, 25), 
              arrowprops=dict(facecolor='darkred', 
                              headlength=5, headwidth=8, width=3))
#%%
x = range(100)
y = [i**2 for i in x]
fig = plt.figure(figsize=(10,8), dpi=300)
plt.plot(x,y)
plt.savefig('save.png')

#%%
# 网格设置
plt.axis([-1, 1, -10, 10])
plt.grid(color='grey', linestyle='--')

#%%
# 图中文字设置
plt.rcParams['font.sans-serif'] = 'SimHei'
fig = plt.figure(figsize=(10,6))
x = range(6)
y = [i**2 for i in x]
plt.plot(x,y)
x_tick = ['零','壹','贰','叁','肆','伍']
plt.xticks(x, x_tick, rotation=90)
plt.grid()
plt.tick_params(labelsize=15, labelcolor='#1f77b4', grid_color='grey')
plt.xlim(left=0, right=5)
plt.ylim(bottom=0, top=25)
plt.xlabel('横轴', fontsize=15)
plt.ylabel('纵轴', fontsize=15)

fig.savefig('图内文字.png')

#%%
# 双y轴
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
fig, ax1 = plt.subplots()
x = np.arange(1e-3, np.e, 0.001)
y1 = np.exp(-x)
y2 = np.log(x)
ax1.plot(x, y1)
ax1.set_ylabel('主轴')
ax1.set_xlabel('横轴')
ax2 = ax1.twinx()
ax2.plot(x, y2)
ax2.set_xlim([0, np.e])
ax2.set_ylabel('次轴')  

#%%
