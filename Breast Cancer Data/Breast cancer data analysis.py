#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np # linear algebra
import pandas as pd
df = pd.read_csv("breast-cancer.csv")
df.head()


# In[42]:


# Information About the data set 
df.info()


# In[43]:


# Import more libraries for presenting good visualizations 

import matplotlib.pyplot as plt
import seaborn as sns


# 
# 
#  ##Attribute Information:
# 
# 1) ID number
# 
# 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# .
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g) concavity (severity of concave portions of the contour)
# 
# h) concave points (number of concave portions of the contour)
# 
# i) symmetry
# 
# j) fractal dimension ("coastline approximation" - 1

# In[44]:


# describe data
df.describe()


# In[45]:


df.diagnosis.value_counts()


# In[46]:


#Missing Values 


# In[47]:


df.isnull().sum()



# In[48]:


df.diagnosis.value_counts()


# In[49]:


df.describe().T


# In[50]:


plt.title('Count of cancer type')
sns.countplot(df, x="diagnosis")
plt.xlabel('Cancer lethality')
plt.ylabel('Count')
plt.show()


# In[51]:


df.hist(figsize = (30,30), color = 'blue')
plt.show()


# In[52]:


X = df.drop(["diagnosis"],axis=1)
y= df['diagnosis']


# In[53]:


X.shape , y.shape


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2)


# In[56]:


from sklearn.tree import DecisionTreeClassifier


# In[57]:


dt = DecisionTreeClassifier(criterion='gini',max_depth=5)
dt.fit(X_train , y_train)


# In[58]:


y_pred = dt.predict(X_test)



# In[59]:


y_pred = pd.DataFrame(y_pred)
y_pred


# In[60]:


from sklearn.metrics import accuracy_score, confusion_matrix , precision_score , recall_score


# In[61]:


accuracy_score(y_test, y_pred)


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


df = pd.DataFrame(df)



# In[66]:





# In[ ]:




