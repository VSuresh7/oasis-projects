#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\WineQT.csv")
df


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[9]:


df['Id']=df['Id'].astype(float)


# In[14]:


df.dtypes


# In[15]:


df.describe()


# In[18]:


df.shape


# In[19]:


df.size


# In[30]:


corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='BuGn')


# In[34]:


quality_counts = df['quality'].value_counts()
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=quality_counts.index, y=quality_counts.values, palette=["blue", "red", "pink"])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points')
plt.title("Variation in wine", fontsize=14)
plt.xlabel("quality", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45)
plt.show()


# In[35]:


quality_counts 


# In[39]:


for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=col, y='quality', data=df)
    plt.title(f'Scatter Plot of Feature {col} vs. quality')
    plt.xlabel(col)
    plt.ylabel('quality')
    plt.show()


# In[40]:


df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# In[43]:


plt.figure(figsize=(15, 8))
df_corr_bar = abs(df.corr()['quality']).sort_values()[:-1]
sns.barplot(x=df_corr_bar.index, y=df_corr_bar.values, palette="Blues_d").set_title('Feature Correlation Distribution According to Quality', fontsize=20)
plt.xticks(rotation=70, fontsize=14)
plt.show()


# In[47]:


for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# In[49]:


df.groupby('quality').mean()


# In[51]:


df['quality'].value_counts()


# In[55]:


df_yn=df.copy()
df_yn["good wine"] = ["No" if index < 6 else "Yes" for index in df_yn['quality']]
X = df_yn.drop(["quality"], axis = 1)
y = df_yn["good wine"]


# In[56]:


y.head()
y.value_counts()


# In[100]:


scaler = MinMaxScaler(feature_range=(0, 1))
normal_df = df_yn.copy()
x = normal_df.drop(columns=['quality'])
normal_df =  normal_df.drop(columns=['quality'])
x


# In[103]:


y = df['quality']
df['good wine']=y
df.head()


# In[115]:


from powerbiclient import *
workspace_id = "dce9e4ef-3946-4c8d-b575-0963cfdd3884"
report_id = "31dab05d-7fc3-4ddf-822d-a698c27c9839"
report = Report(workspace_id=workspace_id, report_id=report_id)
report

