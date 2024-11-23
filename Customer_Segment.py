#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Preparing Dataset for import and cleaning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\ifood_df.csv")
df


# In[3]:


#Making Top 5 values
df.head()


# In[36]:


#Checking Columns data types
df.dtypes


# In[37]:


#Checking Statistical Analysis
df.describe()


# In[38]:


#Comparing columns with data set
df.columns


# In[39]:


#Checking number of rows and columns
df.shape


# In[40]:


#Checking for null values
df.isnull().sum()


# In[41]:


#Number of Unique values
df.nunique()


# In[42]:


df.head()


# In[43]:


#Making a boxplot on Total Amount
sns.boxplot(data=df,y=df['MntTotal'],color='lightgreen',saturation=0.9,)
plt.title('Box Plot of Total AMount')
plt.gca().set_facecolor('orange')
plt.show()


# There are two outliers visible in above box plots.They are represented b two individual data points above upper whisker.The presence of two outliers indicates that there might be some unusual data points that are significantly different from the rest of the data. 

# In[7]:


#Let's check for outliers
Q1 = df['MntTotal'].quantile(0.25)
Q3 = df['MntTotal'].quantile(0.75)
IQR=Q3-Q1
Lower_bound=Q1-1.5*IQR
Upper_bound=Q3+1.5*IQR
outliers=df[(df['MntTotal']<Lower_bound)|(df['MntTotal']>Upper_bound)]
outliers


# In[8]:


#Removal of Outliers
Remove_outlrs = df[(df['MntTotal']>Lower_bound)&(df['MntTotal']<Upper_bound)]
Remove_outlrs.describe()


# In[9]:


#Boxplot for Income
sns.boxplot(data=df,y=df['Income'])
plt.show()


# The whiskers extend to the minimum and maximum values without any significant gaps, indicating that there are no extreme values that would be classified as outliers.

# In[10]:


#Histogram for Income
sns.histplot(data=df, x='Income', bins=30, kde=True)
plt.title('Histogram for Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()


# Here, we can observe that skewness of the graph.Left and Right skewness are close to equal and it will be considered as normal distribution.Hence Income distribution is normal distribution

# In[49]:


df.head(20)


# In[11]:


#Frequency of Fruits Amount
freq=df['MntFruits'].value_counts().head(20)
freq


# We can check here that the amount of fruits is not exceed 35 it is top20 customers data.
# We can use boxplot to get whole data which is frequency of fruits amount

# In[51]:


sns.boxplot(data=df,x=freq)
plt.show()


# Above boxplot we can see frequency of fruits amount.Found data points as outliers

# In[62]:


#Checking for Outliers
Q1=freq.quantile(0.25)
Q3=freq.quantile(0.75)
IQR=Q3-Q1
Lowerbound=Q1-1.5*IQR
Uppperbound=Q3+1.5*IQR
outlierss=freq[(freq<Lowerbound)|(freq>Uppperbound)]
outlierss


# In[58]:


df.dtypes


# In[59]:


corr_demo=['Income']
corr_spends=['MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']


# In[61]:


#Correlation matrix
cor=df[['MntTotal']+corr_demo+corr_spends].corr()
plt.figure(figsize=(5,5))
sns.heatmap(cor,cmap='BuGn',annot=True)
plt.show()


# Here, We can conclude that columns Income and MntMeatProducts are strongly correlated with MntTotal.The Correlation between "MntTotal" and "Income" is 0.82 hence when a customer has more income then he can spends.

# In[83]:


#Scattering all data points of cols income and MntTotal
plt.scatter(df['Income'],df['MntTotal'])
plt.show()


# In[84]:


#Using KMeans Clustering here data points changing to clusters based on patterns or distance value
km=KMeans(n_clusters=3)
km


# In[78]:


clust=km.fit_predict(df[['Income','MntTotal']])
clust


# In[79]:


df['clusters']=clust


# In[80]:


data_table = df[['Income','MntTotal','clusters']]
df['clusters'].value_counts()


# In[81]:


df1=df[df['clusters']==0]
df2=df[df['clusters']==1]
df3=df[df['clusters']==2]
plt.scatter(df1['Income'],df1['MntTotal'])
plt.scatter(df2['Income'],df2['MntTotal'])
plt.scatter(df3['Income'],df3['MntTotal'])

plt.show()


# This graph displays the results of K-Means clustering applied to a dataset with two cols Income and  MntTotal.There are three clusters have been identified and each represented by green,blue and orange.The cluster appear to be well-Separeted suggesting that K-Means algorithm has effectively identified meaningfull groupings within data.

# In[3]:


#Standardizing Data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cls_clustering= ['Income','MntTotal','Kidhome']
data_scaled=df.copy()
data_scaled[cls_clustering]=scaler.fit_transform(df[cls_clustering])
data_scaled[cls_clustering].describe()


# In The above stats we can observe that mean of both cols is near to zero and standard deviation is one.hence data is standardized.

# In[90]:


#Elbow Method
X=data_scaled[cls_clustering]
inertia_list=[]
for K in range(2,10):
    inertia=KMeans(n_clusters=K,random_state=7).fit(X).inertia_
    inertia_list.append(inertia)


# In[95]:


plt.plot(inertia_list)
plt.title("Inertia Elbow Method")
plt.show()


# This plot suggests that two clusters k=2 might be the optimal number .The Elbow point is aroun 2. Lets' Check Silhouette Score

# In[19]:


#Princple Component Analysis
from sklearn import decomposition
pca= decomposition.PCA(n_components=2)
pca_res = pca.fit_transform(data_scaled[cls_clustering])
data_scaled['pc1']=pca_res[:,0]
data_scaled['pc2']=pca_res[:,1]


# In[93]:


#Silhouette_score
from sklearn.metrics import silhouette_score
silhouette_list=[]
for K in range(2,10):
    model=KMeans(n_clusters=K,random_state=7).fit_predict(X)
    sil_avg =silhouette_score(X,model)
    silhouette_list.append(sil_avg)


# In[96]:


plt.plot(silhouette_list)
plt.show()


# This plots suggests that a lower number of clusters might be more appropriate for this dataset.As the number of clusters increases,the silhouette score decreases, indicating less well-defined clusters.
# Combining both plots,it seem that having two clusters might be a good choice for this dataset (K=2). 

# In[16]:


model=KMeans(n_clusters=2,random_state=7)
model.fit(data_scaled[cls_clustering])
data_scaled['cluster']=model.predict(data_scaled[cls_clustering])


# In[31]:


#Data Visualization
sns.scatterplot(x='pc1',y='pc2',data=data_scaled,hue='cluster',palette='Set1')
plt.gca().set_facecolor('gray')
plt.title('Clustered Data Visualization')
plt.show()


# In[32]:


data_scaled['cluster'].value_counts()


# In[42]:


df['cluster']=data_scaled.cluster
df.groupby('cluster')[cls_clustering].mean()


# In[50]:


#Products mean by clusters
product_amnt=['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','MntRegularProds']
avg_amnt_cluster=df.groupby('cluster')[product_amnt].mean().reset_index()
avg_amnt_cluster.head()


# In[82]:


melted_data=pd.melt(df,id_vars='cluster',var_name='products',value_name='consumption')
sns.barplot(x='cluster',y='consumption',data=melted_data,hue='products',palette='viridis',ci=None,width=0.7,saturation=0.9)
plt.title('product consumption by cluster')
plt.legend(handlelength=0.8)
plt.show()


# In[103]:


#cluster size
cluster_size=df.groupby('cluster')[['MntTotal']].count().reset_index()
sns.barplot(x='cluster',y='MntTotal',data=cluster_sizes,width=.5,saturation=0.4,palette='hot')
plt.title("cluster Size")
plt.show()


# In[107]:


total_rows=len(df)
cluster_size['Share%']=round(cluster_size['MntTotal']/total_rows*100,0)
cluster_size


# In[126]:


#Share of the clusters
plt.pie(cluster_size['Share%'],labels=cluster_size.index,autopct='%1.1f%%',colors=['beige','gray'],wedgeprops={'edgecolor':'black'})
plt.legend(title='clusters')
plt.show()


# In[129]:


#Income by clusters 
sns.boxplot(x='cluster',y='Income',data=df)
plt.title('Income by Clusters')
plt.show()


# In[136]:


sns.scatterplot(x='cluster',y='Income',data=df,hue='cluster',palette='magma')
plt.title('Income by  clusters')
plt.show()


# In[140]:


sns.scatterplot(x='cluster',y='MntTotal',data=df,hue='cluster')
plt.show()


# # RESULTS

# The results of the K-Means clustering analysis,that pointed to identify different customer segments based on the total amount of purchases by MntTotal.In this analysis I've used Income and kidhome columns.

# # Cluster

# According to Elbow method and Silhouette Analysis suggested it is better to use 2 cluster for this data set,hence k=2.

# # Recommendations

# Based on the clusters, We can obersved that
# Cluster 0 is characterized by higher income levels and elevated spending on diverse products, in contrast to Cluster 1, which is characterized by lower income levels and redeced spending.
# based on this,
# Cluster 0: High-Income,High-Spending Customers:
# Implement personalize marketing campaigns that highlight premium products and exclusive offer and Develop a tiered loyalty program
# Cluster 1:Low-Income,Low-Spending Customers:
# Focus on promotions,discounts and budget-friendly products,offer simple loyalty programs with rewards for repeat purchaes and Create affordable product bundles to increase average order value

# In[142]:


#Integrating Powerbi dashboard with jupyter notebook
from powerbiclient import *
workspace_id = "dce9e4ef-3946-4c8d-b575-0963cfdd3884"
report_id = "449c40e9-07ca-49c2-b69b-44ddf7c9dc1a"
report = Report(workspace_id=workspace_id, report_id=report_id)
report

