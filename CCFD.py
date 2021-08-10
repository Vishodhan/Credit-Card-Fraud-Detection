#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[2]:


data = pd.read_csv("creditcard.csv")


# In[3]:


data.head()


# In[4]:


print(data.shape)
print(data.describe())


# In[5]:


data.hist(figsize = (20, 20))
plt.show()


# In[6]:


fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(normal))
print("Outlier %age : "+ str(outlierFraction))
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[7]:


print("Amount details of the fraudulent transaction")
fraud.Amount.describe()


# In[8]:


print("details of valid transaction")
normal.Amount.describe()


# In[9]:


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[10]:


X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
xData = X.values
yData = Y.values
print(Y)


# In[11]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)
estimator = rfc.estimators_[5]


# In[16]:


#visualizing the random tree 
feature_list = list(X.columns)
# Import tools needed for visualization
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
import os

#pulling out one tree from the forest
export_graphviz(estimator, out_file = 'tree.dot', feature_names = list(X.columns), rounded = True, precision = 1)
# Use dot file to create a graph
(graph) = pydotplus.graph_from_dot_file('tree.dot')
# Write graph to a png file
# graph.create_png('tree.png')
Image(graph.create_png())


# In[19]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")

acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))

f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is {}".format(MCC))


# In[18]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS,yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

