#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("DATASET-B.csv")
len(df)


# In[5]:


# df.sample(10, random_state=233)


# In[6]:


# features = list(df.columns)[5:-2]
features = ['aveSpeed', 'gridAcc', 'volume', 'speed_std', 'stopNum']
# df[features].describe()


# In[7]:


sns.countplot(x="labels", data=df)
plt.show()


# In[8]:


plt.figure(dpi=200)
corr = df[features + ["labels"]].corr()
sns.heatmap(corr, square=True)
plt.show()


# In[9]:


fig, ax = plt.subplots(3, 2, figsize=(10, 3.5), dpi=200)
for axi, feature in zip(ax.ravel(), features):
    sns.boxplot(x=feature, y='labels', orient='h', data=df, fliersize=1, ax=axi)
plt.tight_layout()
plt.show()


# In[10]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[11]:


df_sample = df.sample(20000, random_state=233)
X = df_sample[features]
y = df_sample["labels"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=233)


# In[12]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


params = {
    "kernel": ["rbf"],
    "gamma": [0.1, 0.2, 0.5, 1],
    "C": [10, 100, 1e3]
}
clf = GridSearchCV(SVC(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.best_params_)


# In[14]:


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

