#!/usr/bin/env python
# coding: utf-8

# In[130]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\Housing.csv")
df


# In[3]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df.info()


# In[26]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(df['price'], ax = axs[0,0])
plt2 = sns.boxplot(df['area'], ax = axs[0,1])
plt3 = sns.boxplot(df['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(df['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(df['stories'], ax = axs[1,1])
plt3 = sns.boxplot(df['parking'], ax = axs[1,2])
axs[0,0].set_xlabel('price')
axs[0,1].set_xlabel('area')
axs[0,2].set_xlabel('bedrooms')
axs[1,0].set_xlabel('bathrooms')
axs[1,1].set_xlabel('stories')
axs[1,2].set_xlabel('parking')

plt.tight_layout()


# In the above boxplots, we can consider outliers of price and area.Several data points plotted above the upper whisker, representing values significantly larger than the majority of the data

# In[45]:


#Outliers for Price
plt.boxplot(df['price'])
Q1=df['price'].quantile(0.25)
Q3=df['price'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
house=df[(df['price']>=lower_bound)&(df['price']<=upper_bound)]


# In[43]:


#Outliers for area
plt.boxplot(df['area'])
Q1=df['area'].quantile(0.25)
Q3=df['area'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
House_area=df[(df['area']>=lower_bound)&(df['area']<=upper_bound)]


# In[53]:


#outliers analysis
fig,axs=plt.subplots(2,3,figsize=(10,7))
plt1=sns.boxplot(df['price'],ax=axs[0,0])
plt2=sns.boxplot(df['area'],ax=axs[0,1])
plt3=sns.boxplot(df['bedrooms'],ax=axs[0,2])
plt1=sns.boxplot(df['bathrooms'],ax=axs[1,0])
plt2=sns.boxplot(df['stories'],ax=axs[1,1])
plt3=sns.boxplot(df['parking'],ax=axs[1,2])
plt.tight_layout()


# In[63]:


sns.pairplot(df,hue='price',palette='viridis',diag_kind='hist')
plt.show()


# In[64]:


df.head()


# In[84]:


fig,axs = plt.subplots(2,3,figsize=(20,15))
sns.boxplot(x='mainroad',y='price',data=df,ax=axs[0,0])
sns.boxplot(x='guestroom',y='price',data=df,ax=axs[0,1])
sns.boxplot(x='basement',y='price',data=df,ax=axs[0,2])
sns.boxplot(x='hotwaterheating',y='price',data=df,ax=axs[1,0])
sns.boxplot(x='airconditioning',y='price',data=df,ax=axs[1,1])
sns.boxplot(x='furnishingstatus',y='price',data=df,ax=axs[1,2],hue='airconditioning')
plt.show()


# In[83]:


sns.boxplot(x='furnishingstatus',y='price',data=df,hue='airconditioning')
plt.show()


# In[2]:


df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\Housing.csv")
df.head()


# In[29]:


bool_list=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[bool_list]=df[bool_list].replace({'yes':1,'no':0})
df


# In[30]:


df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\Housing.csv")
df


# In[34]:


bool_list=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[bool_list]=df[bool_list].replace({'yes':1,'no':0})
df.head()


# In[36]:


furnshish_stats=['furnishingstatus']
df[furnshish_stats]=df[furnshish_stats].replace({'furnished':0,'semi-furnished':1,'unfurnished':2})
df


# In[37]:


df.dtypes


# In[82]:


bool_list=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[bool_list]=df[bool_list].replace({'yes':1,'no':0})
df.head()


# In[83]:


furnshish_stats=['furnishingstatus']
df[furnshish_stats]=df[furnshish_stats].replace({'furnished':0,'semi-furnished':1,'unfurnished':2})
df


# In[84]:


np.random.seed(0)
df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3,random_state=100)


# In[85]:


scaler = MinMaxScaler()
num_cols=['price','area','bedrooms','bathrooms','stories','parking']
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])


# In[86]:


df_train


# In[87]:


df_train.describe()


# In[89]:


plt.figure(figsize = (15, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[90]:


pair=['price','area']
sns.pairplot(df[pair],palette='viridis')


# In[91]:


y_train = df_train.pop('price')
X_train = df_train


# In[92]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[96]:


rfe = RFE(estimator=lm, n_features_to_select=6)
rfe = rfe.fit(X_train, y_train)


# In[97]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[98]:


col = X_train.columns[rfe.support_]
col


# In[99]:


X_train.columns[~rfe.support_]


# In[100]:


X_train_rfe = X_train[col]


# In[103]:


X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()
lm.summary()


# In[105]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[106]:


y_train_price = lm.predict(X_train_rfe)
res = (y_train_price - y_train)


# In[107]:


fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   


# In[108]:


plt.scatter(y_train,res)
plt.show()


# In[120]:


num_vars = ['price','area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[124]:


df_test['price']=df['price']


# In[125]:


df_test.columns


# In[126]:


y_test = df_test.pop('price')
X_test = df_test


# In[127]:


X_test = sm.add_constant(X_test)


# In[128]:


X_test_rfe = X_test[X_train_rfe.columns]


# In[129]:


y_pred = lm.predict(X_test_rfe)


# In[131]:


r2_score(y_test, y_pred)


# In[132]:


fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16)     


# we can see here equation best fits : Price = 0.35*area + 0.20*bathrooms + 0.19*stories+ 0.10*airconditioning + 0.10*parking + 0.11*prefarea
