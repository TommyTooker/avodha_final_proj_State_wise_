#!/usr/bin/env python
# coding: utf-8

# # Project objective
# 
# In this project Iam comparing  population, education, growth development accross states based on the 2001 and 2011-12 data. Gender-based ratio unemployment and literacy how the variables effecting the current environment.

# # About data
# 
# This datasets contains data from RBI which is published annually and this data has different features such as
# Column names and legends
# 
# 2000-01-INC = Income of each state for the year 2001
# 2011-12-INC = Income of each state for the year 2011
# 
# 2001 - LIT = Literacy rate of each state for the year 2001
# 2011- LIT = Literacy rate of each state for the year 2011
# 
# 2001 - POP = Total population of each state for the year 2001
# 2011- POP = Total population of each state for the year 2011
# 
# 2001 -SEX_Ratio = Sex_Ratio of the each state for the year 2001
# 2011 -SEX_Ratio = Sex_Ratio of the each state for the year 2011
# 
# 2001 -UNEMP = Unemployment rate of the each state for the year 2001
# 2011 -UNEMP = Unemployment rate of the each state for the year 2011
# 
# 2001 -Poverty = Poverty rate of the each state for the year 2001
# 2011 -Poverty = Poverty rate of the each state for the year 2001
# 

# # Loading data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\acer\Downloads\archive(1)\RBI DATA states_wise_population_Income.csv")


# # Basic cleaning and preprocessing

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe().T


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# ### This data does not have any NULL OR DUPLICATE values

# # EDA

# In[9]:


states_best_income = df[['States_Union Territories','2000-01-INC','2011-12-INC']]
states_best_income.sort_values(by = '2011-12-INC',ascending=False,ignore_index=True).head(5)


# In[11]:


states_best_income.plot(kind='bar',x='States_Union Territories')
plt.title('Income comparison')
plt.show()


# In[12]:


states_best_literacy = df[['States_Union Territories','2001 - LIT','2011- LIT']]
states_best_literacy.sort_values(by = '2011- LIT',ascending=False,ignore_index=True).head(5)


# In[13]:


states_best_literacy.plot(kind='bar',x='States_Union Territories')
plt.title('Literacy comparison')
plt.show()


# In[14]:


states_best_poverty = df[['States_Union Territories','2001 -Poverty','2011 -Poverty']]
print(states_best_poverty.sort_values(by = '2011 -Poverty',ascending=False,ignore_index=True).head(5))
print(states_best_poverty.sort_values(by = '2011 -Poverty',ascending=False,ignore_index=True).tail(5))


# In[15]:


states_best_poverty.plot(kind='bar',x='States_Union Territories')
plt.title('poverty comparison')
plt.show()


# In[16]:


#States Income Vs States Literacy rate
#States Poverty rate vs States Literacy rate

#subplot 1: Income

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.plot(df['States_Union Territories'],df['2000-01-INC'],)
plt.plot(df['States_Union Territories'],df['2011-12-INC'])
plt.xticks(rotation = 90)
plt.xlabel('States')
plt.ylabel('Income')
plt.title('State wise Income')
plt.show()

#subplot 2: Literasy
plt.figure(figsize=(12,6))
plt.subplot(2,2,2)
plt.plot(df['States_Union Territories'],df['2001 - LIT'])
plt.plot(df['States_Union Territories'],df['2011- LIT'])
plt.xticks(rotation = 90)
plt.title('State wise Literasy')
plt.xlabel('States')
plt.ylabel('Literacy')
plt.show()

#subplot 3: poverty

plt.figure(figsize=(12,6))
plt.subplot(2,2,3)
plt.plot(df['States_Union Territories'],df['2001 -Poverty'])
plt.plot(df['States_Union Territories'],df['2011 -Poverty'])
plt.xticks(rotation= 90)
plt.xlabel('State')
plt.ylabel('Poverty')
plt.title('State wise poverty')
plt.show()
          


# In[17]:


states_best_income = df[['States_Union Territories','2000-01-INC','2011-12-INC','2001 -Poverty','2011 -Poverty']]
states_best_income.sort_values(by = '2011 -Poverty',ascending=True,ignore_index=True).head(5)


# In[18]:


#States Income vs States Poverty rate


fig, ax1 = plt.subplots(figsize=(8, 8))
X = df['States_Union Territories']
y = df['2011-12-INC']
z = df['2011 -Poverty']
ax2 = ax1.twinx()
ax1.plot(X,y,color = 'r')
ax2.plot(X,z,color='g')
ax1.tick_params(axis='x', labelrotation = 90)
ax1.set_xlabel('States')
ax2.set_ylabel('Poverty',color = 'r')
ax1.set_ylabel('Income',color = 'g')


# ### - In between 2001 to 2011 all the  states has slightly shoot up there  poverty level
# ### - goa has shown a dominating income,poverty field 

# In[19]:


plt.figure(figsize=(16,6))
plt.plot(df['States_Union Territories'],df['2001 -SEX_Ratio'])
plt.plot(df['States_Union Territories'],df['2011 -SEX_Ratio'])
plt.title('States wise sex_rstio')
plt.xlabel('state')
plt.ylabel('sex_ratio')
plt.xticks(rotation=90)
plt.show()


# In[20]:


unemployment = df[['States_Union Territories','2001 -UNEMP','2011 -UNEMP']]
unemployment.sort_values(by = '2011 -UNEMP',ignore_index=True,ascending=False).head()


# In[21]:


sns.lineplot(df['States_Union Territories'],df['2001 -UNEMP'])
sns.barplot(df['States_Union Territories'],df['2011 -UNEMP'])
plt.xticks(rotation =90)
plt.title('States wise unemployment')
plt.show()


# In[22]:


population = df[['States_Union Territories','2001 - POP','2011- POP']]

population.sort_values(by = '2011- POP',ascending=False,ignore_index=True).head()


# In[23]:


plt.figure(figsize=(10,10))
sns.pairplot(df,diag_kind='kde')
plt.show()


# In[24]:


df_corr = df.corr()
df_corr


# In[25]:


plt.figure(figsize=(12,8))
sns.heatmap(df_corr,cmap ='crest',annot=True)


# ### - income,literacy feilds are comparatively improved at 2011 
# 
# 

# ### - Income poverty, income population both features has some negative corelation
# ### - income literacy and income unemployment has above 0.50 relationship 

# # Model slection and Machine learning 

# In[26]:


df_ml = df[['2011-12-INC','2011- LIT','2011- POP','2011 -UNEMP','2011 -Poverty']]


# In[27]:


df_ml.head()


# In[28]:


df_ml.corr()


# In[29]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


# In[30]:


s_scale = StandardScaler()


# In[31]:


ss_df = pd.DataFrame(s_scale.fit_transform(df_ml),columns=df_ml.columns)


# In[32]:


ss_df.head()


# ## Linear regression

# In[33]:


x = ss_df.drop('2011-12-INC',axis =1)
x.head()


# In[34]:


y = pd.DataFrame(ss_df['2011-12-INC'])
y.head()


# In[35]:


lr = LinearRegression()


# In[36]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=32)


# In[37]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[38]:


lr.fit(x_train,y_train)


# In[39]:


lr.score(x_train,y_train)


# ### Train score is 0.43

# In[40]:


lr.score(x_test,y_test)


# ### Test score is 0.015

# In[41]:


pred = lr.predict(x_test)
print('r2 value of train set is',r2_score(y_test,pred))


# # K_means

# In[42]:


kmean = KMeans(n_clusters = 4,random_state = 1)

kmean.fit(ss_df)


# In[43]:


kmeans_label = kmean.labels_


# In[44]:


ss_df['kmeans_label']=kmeans_label


# In[45]:


ss_df.head()


# In[46]:


ss_df['kmeans_label'].value_counts()


# In[47]:


ss_df['State_name'] = df['States_Union Territories']


# In[48]:


ss_df.head()

