#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\Extracted\\datasets\\apps.csv")
df


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.size


# In[11]:


df.isnull().sum()


# Here we can replace null values by filling preceding/succeeding non-null values.

# In[16]:


cols_to_replace=['Rating','Size','Android Ver','Current Ver']
df[cols_to_replace]=df[cols_to_replace].fillna(method='bfill')
df.isnull().sum()


# In[18]:


df.dtypes


# In[54]:


chars_to_remove = [',','$','+']
cols_to_clean = ['Installs','Price']
for col in cols_to_clean:
    for char in chars_to_remove:
        df[col] = df[col].astype(str).str.replace(char,'')
    df[col] = pd.to_numeric(df[col])  


# Google Play continues to be an important distribution platform to build a global audience. For businesses to get their apps in front of users, it's important to make them more quickly and easily discoverable on Google Play. To improve the overall search experience

# In[27]:


import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
num_categories = len(df['Category'].unique())
print('Number of categories = ', num_categories)
num_apps_in_category = df['Category'].value_counts().sort_values(ascending = False)

data = [go.Bar(
        x = num_apps_in_category.index, 
        y = num_apps_in_category.values,
)]

plotly.offline.iplot(data)


# We will see that there are 33 unique app categories present in our dataset. Family and Game apps have the highest market prevalence. Interestingly, Tools, Business and Medical apps are also at the top.

# In[30]:


avg_app_rating = df['Rating'].mean()
print('Average app rating = ', avg_app_rating)
data = [go.Histogram(
        x = df['Rating']
)]
layout = {'shapes': [{
              'type' :'line','x0': avg_app_rating,'y0': 0,'x1': avg_app_rating,'y1': 1000,'line': { 'dash': 'dashdot'}}]
          }

plotly.offline.iplot({'data': data, 'layout': layout})


# From our research, we found that the average volume of ratings across all app categories is 4.17. The histogram plot is skewed to the left indicating that the majority of the apps are highly rated with only a few exceptions in the low-rated apps.

# If the mobile app is too large, it may be difficult and/or expensive for users to download. Lengthy download times could turn users off before they even experience your mobile app. Plus, each user's device has a finite amount of disk space. For price, some users expect their apps to be free or inexpensive. These problems compound if the developing world is part of your target market; especially due to internet speeds, earning power and exchange rates.

# In[44]:


sns.set_style("dark")
apps_with_size_and_rating_present = df[(~df['Rating'].isnull()) & (~df['Size'].isnull())]
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()
plt1 = sns.jointplot(x = large_categories['Size'], y = large_categories['Rating'], kind = 'hex')
paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type'] == 'Paid']
plt2 = sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'])


# In[56]:


df['Price']=df['Price'].astype(float)
df.dtypes


#  Relation between app category and app price

# In[57]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
popular_app_cats = df[df.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY','MEDICAL', 'TOOLS', 'FINANCE','LIFESTYLE','BUSINESS'])]
ax = sns.stripplot(x = popular_app_cats['Price'], y = popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')
apps_above_200 = popular_app_cats[['Category', 'App', 'Price']][popular_app_cats['Price'] > 200]
apps_above_200


# Filter out "junk" apps

# In[58]:


apps_under_100 = popular_app_cats[popular_app_cats['Price']<100]
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
ax = sns.stripplot(x=apps_under_100['Price'], y=apps_under_100['Category'], data=apps_under_100,jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories after filtering for junk apps')


# Popularity of paid apps vs free apps

# In[65]:


cat1 = go.Box(
    y=df[df['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

cat2 = go.Box(
    y=df[df['Type'] == 'Free']['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(
        type = 'log',
        autorange = True
    )
)
data = [cat1, cat2]
plotly.offline.iplot({'data': data, 'layout': layout})


# Sentiment analysis of user reviews

# In[68]:


reviews_df = pd.read_csv("C:\\Users\\Vadty\\OneDrive\\Desktop\\DA\\Extracted\\datasets\\user_reviews.csv")
merged_df = pd.merge(df, reviews_df, on = 'App', how = "inner")
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
ax = sns.boxplot(x = merged_df['Type'], y = merged_df['Sentiment_Polarity'], data = merged_df)
ax.set_title('Sentiment Polarity Distribution')

