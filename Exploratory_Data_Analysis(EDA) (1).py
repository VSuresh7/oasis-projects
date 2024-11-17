#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Data Loading and Cleaning: Load the retail sales dataset.
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\retail_sales_dataset.csv")
data


# In[16]:


data.head()


# In[17]:


data.isnull().sum()


# In[18]:


data.dtypes


# In[20]:


data['Product Category'].unique()


# In[ ]:





# In[21]:


# Calculate basic statistics (mean, median, mode, standard deviation)
data['Total Amount'].mean()


# In[4]:


data['Total Amount'].median()


# In[5]:


data['Total Amount'].mode()


# In[6]:


data['Total Amount'].std()


# In[22]:


data.describe()


# #Note:
# On the above decription we can Conclude that Qunatity of the products mostly is 4.It means that customers are bought products in more quantity they are not interested to buy in a single quantity only few of them are going with smaller quatity.
# #Note2:
# Most of the customers are making their bill amount is more than 500.It means some of our customers are interested to buy products at our market.

# In[8]:


#Converting Date to datetime
data['Date']=pd.to_datetime(data['Date'])
data.dtypes


# In[72]:


#Time Series
df=data.head(10)
plt.plot(df['Date'],df['Total Amount'],color='green')
plt.xticks(rotation = 35)
plt.yticks(rotation = 35)
plt.gca().set_facecolor('lightgray')
plt.show()


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
data = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\PythonProject\\retail_sales_dataset.csv")
df = data.head(20)
df = df.assign(rolling_mean=df['Total Amount'].rolling(window=6).mean())
plt.plot(df['Date'], df['Total Amount'], label='Total Sales')
plt.plot(df['Date'], df['rolling_mean'], label='Rolling Mean', color='red')
plt.legend(loc='upper left')
plt.xticks(rotation=35)
plt.gca().set_facecolor('pink')
plt.show()
result = seasonal_decompose(df['Total Amount'], model='additive', period=8)
result.plot()
plt.show()
model = ARIMA(df['Total Amount'], order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)


# In[13]:


data.head(20)


# In[5]:


#What is age distribution of a customer
data['Age'].value_counts().sort_values(ascending=True)


# In[67]:


data['Gender'].value_counts()


# In[6]:


#Average Age of customers
#data['Age'].mean()
average_age = data['Age'].mean()
print(f"The average age of customers is {average_age:.3f} years.")


# In[15]:


#Which product category is the most popular among male customers?
most_male=data[data['Gender']=='Male']
Product=most_male['Product Category'].mode()[0]
print("The most popular Product among male customer is",Product)


# In[28]:


#Which age group spends the most on electronics?
Category = data[data['Product Category']=='Electronics']
Category['Age Group']=pd.cut(Category['Age'],bins=[18,30,42,55,60],labels=['18-29','30-41','42-54','55+'])
customers_spending=Category.groupby('Age Group')['Total Amount'].sum()
customers_spending


# In[6]:


#Which product category has the highest average purchase amount?
avg_amount=data.groupby('Product Category')['Total Amount'].mean()
Category=avg_amount.idxmax()
print(f"The Highest average purchase amount of Product Category is '{Category}'")


# In[5]:


#Identify frequent buyers (customers who have made more than 1 purchases).
buyers=data.groupby('Customer ID')['Transaction ID'].count().reset_index()
frequent_buyers=buyers[buyers['Transaction ID']>=1]
frequent_buyers


# In[18]:


#Segment customers based on their average purchase amount (e.g., high-value, medium-value, low-value).


# In[2]:


#Analyze customer demographics
gp = data.groupby(data['Gender']).agg({'Gender':'count'})
plt.pie(gp['Gender'],labels=gp.index,autopct='%1.1f%%')
plt.title('Gender Analysis of Customer')
plt.show()


# In[45]:


gndr=data[(data['Age']>50) & (data['Product Category']=='Beauty')&(data['Quantity']==3)]
gndr
group = gndr.groupby(gndr['Gender']).agg({'Gender':'count'})
plt.pie(group['Gender'],labels=group.index,autopct='%1.2f%%')
plt.title('Gender analysis with conditions')
plt.show()
gndr['Gender'].value_counts()


# In[26]:


gndr['Gender'].value_counts()


# In[64]:


#What is the distribution of sales across different product categories?
import matplotlib.pyplot as plt
sales = data.groupby('Product Category')['Total Amount'].sum()
sales.plot(kind='bar',width=0.3,color='black',edgecolor='white')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.gca().set_facecolor('lightgray')
plt.show()


# In[44]:


#Which product category has the highest average price per unit?
sales=data.groupby('Product Category')['Price per Unit'].mean()
product = sales.idxmax()
print(f'the highest average price per unit of product category is "{product}"')


# In[61]:


#Which customer segment (based on age) spends the most on electronics?
data['Product Category']=='Electronics'
data['Age_group']=pd.cut(data['Age'],bins=[18,25,35,43,55],labels=['18-24','25-34','35-54','55+'])
df=data.groupby('Age_group')['Total Amount'].sum().sort_values(ascending=False)
plt.pie(df,labels=df.index,autopct='%1.2f%%')
plt.title('CUSTOMER SEGMENTS SPENDS THE MOST ON ELECTRONICS')
plt.show()


# In[24]:


data['Date']=pd.to_datetime(data['Date'])


# In[25]:


data.dtypes


# In[26]:


data.head()


# In[34]:


#How does the total sales amount change over time?
sales_amount=data.groupby(data['Date'].dt.to_period('m'))['Total Amount'].sum()
plt.pie(sales_amount,labels=sales_amount.index,autopct='%1.1f%%')
plt.show()


# In[62]:


#Which gender spends more on beauty products?
data[data['Product Category']=='Beauty']
gender_location=data.groupby('Gender')['Product Category'].count()
spends_gender=gender_location.idxmax()
print(f'Gender spends more on beauty products is "{spends_gender}"')


# In[82]:


#Which age group is most likely to buy clothing?
data[data['Product Category']=="Clothing"]
data['Age_Group']=pd.cut(data['Age'],bins=[18,30,40,50,55],labels=['18-29','30-39','40,49','50+'])
group = data.groupby('Age_Group')['Product Category'].count()
group.head(1)


# In[61]:


#What is the average purchase frequency for customers of different genders
avg_purchase=data.groupby('Gender')['Total Amount'].mean()
avg_purchase.plot(kind='bar',edgecolor='black',color='orange',width=0.1)
plt.gca().set_facecolor('lightgreen')
plt.show()


# In[6]:


data.head()


# In[59]:


#Create a bar chart to visualize the distribution of customers by gender.
gender_distribution=data.groupby('Gender')['Total Amount'].sum()
gender_distribution.plot(kind='bar',width=0.2,label='Total Amount spent',color=['Red','Orange'])
plt.legend(loc='upper center')
plt.ylabel('Total Amount')
plt.gca().set_facecolor('gray')
plt.show()


# In[54]:


# Create a bar chart to compare the total sales for each product category.
total_sales=data.groupby('Product Category')['Total Amount'].sum()
total_sales.plot(kind='bar',color=['Red','Orange','green'],edgecolor='black')
plt.gca().set_facecolor('lightblue')
plt.show()


# In[85]:


#Create a bar chart to show the top 5 customers based on their total purchase amount
top_5=data.groupby('Customer ID')['Total Amount'].sum()
only_5=top_5.sort_values(ascending=False)
only_5.head().plot(kind='bar',edgecolor='white')
plt.gca().set_facecolor('lightblue')
plt.ylabel('Total Amount spent')
plt.show()


# In[38]:


from powerbiclient import *
workspace_id = "dce9e4ef-3946-4c8d-b575-0963cfdd3884"
report_id = "4406779a-b880-4655-b3db-cb82dd0cce00"
report = Report(workspace_id=workspace_id, report_id=report_id)
report

