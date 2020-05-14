#!/usr/bin/env python
# coding: utf-8

# ## needed packages

# In[460]:


import pandas as pd
import numpy as np 
import random as rnd

#visiualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ## load our data

# In[461]:


training_path='E:\\Machine_learning_projects\\titanic\\train.csv'
test_path='E:\\Machine_learning_projects\\titanic\\test.csv'
train=pd.read_csv(training_path)
test=pd.read_csv(test_path)
combined=[train,test]


# In[462]:


train.columns


# In[463]:


train.head()


# In[464]:


#check num of missing values
print(train.info())
print('-'*40)
print(test.info())


# ## observation
#      

# In training set : 
#   age,cabin and Embarked have missing values
# In test set: 
#    age,fare and cabin have missing values

# In[465]:


train.describe()


# In[466]:


#to get describe od objects 
train.describe(include=['O'])


# ## observation

# most of passenger were males                       
# most of the passenger travelled from tha same port

# In[389]:


train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[390]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# most of survived passengers were from class 1       
# most of survived passengers were womans

# ## Analyzing by plotting 

# In[391]:


plt.figure(figsize=(18,8))
train['Survived'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.title('Survived')


# In[392]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived vs dead')
plt.show()


# In[393]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('num of passengers by pclass')
ax[0].set_ylabel('count')
sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('pclass vs survived vs dead')
plt.show()


# In[394]:


plt.figure(figsize=(14,12))
g=sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)


# In[395]:


plt.figure(figsize=(14,12))
g=sns.FacetGrid(train,col='Survived',row='Pclass')
g.map(plt.hist,'Age',bins=20)


# In[396]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Embarked'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('num of passengers by pclass')
ax[0].set_ylabel('count')
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('pclass vs survived vs dead')
plt.show()


# ## cleaning data

# In[397]:


train=train.drop(['Cabin','Ticket'],axis=1)
test=test.drop(['Cabin','Ticket'],axis=1)


# In[398]:


train.shape,test.shape


# In[399]:


combine=[train,test]


# In[400]:


train['Title']=train['Name'].str.extract('([A-Za-z]+)\.',expand=False)   
test['Title']=test['Name'].str.extract('([A-Za-z]+)\.',expand=False)   
pd.crosstab(train.Title,train.Sex)


# In[401]:


train['Title']=train['Title'].replace(['Capt','Col','Don','Countess','Dr','Jonkheer','Lady','Major','Sir','Rev'],'rare')
test['Title']=test['Title'].replace(['Capt','Col','Countess','Dr','Jonkheer','Lady','Major','Sir','Rev'],'rare')


# In[402]:


train['Title']=train['Title'].replace(['Mlle','Ms','Mme'],'Miss')
test['Title']=test['Title'].replace(['Mlle','Ms','Mme'],'Miss')


# In[403]:


train.head()
train[['Title','Survived']].groupby('Title',as_index=False).mean()


# In[404]:


title_mapping={'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rare':5}
train['Title']=train['Title'].map(title_mapping)
train['Title']=train['Title'].fillna(0)
test['Title']=test['Title'].map(title_mapping)
test['Title']=test['Title'].fillna(0)
train['Title']=train['Title'].astype(int)
train['Title']=train['Title'].astype(int)


# In[405]:


train.head()


# In[406]:


train=train.drop(['PassengerId','Name'],axis=1)


# In[407]:


test=test.drop(['PassengerId','Name'],axis=1)


# In[408]:


guess_age=np.zeros((2,3))


# In[409]:


train['Sex']=train['Sex'].map({'female':0,'male':1})


# In[410]:


train['Sex']=train['Sex'].astype(int)


# In[411]:


test['Sex']=test['Sex'].map({'female':0,'male':1}).astype(int)


# In[412]:


train.head()


# In[413]:


combined=[train,test]
for dataset in combined:
    for i in range (0,2):
        for j in range (0,3):
            guess_df=dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()
            guess=guess_df.median()
            guess_age[i,j]=guess
    
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset['Sex']==i)&(dataset['Pclass']==j+1),'Age']=guess_age[i,j]
    
    
    dataset['Age']=dataset['Age'].astype(int)
            
            
train.info()


# In[414]:


train['AgeRange']=pd.cut(train['Age'],5)


# In[415]:


train[['AgeRange','Survived']].groupby('AgeRange',as_index=False).mean().sort_values(by='AgeRange')


# In[416]:


for dataset in combined:
    dataset.loc[dataset['Age']<=16,'Age']=1
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=2
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=3
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=4
    dataset.loc[(dataset['Age']>64)&(dataset['Age']<=80),'Age']=5
    dataset['Age']=dataset['Age'].astype(int)


# In[417]:


train.head()


# In[418]:


train=train.drop('AgeRange',axis=1)


# In[419]:


combined=[train,test]
for dataset in combined:
    dataset['familysize']=dataset['SibSp']+dataset['Parch']+1
    
train[['familysize','Survived']].groupby('familysize',as_index=False).mean().sort_values(by='familysize')


# In[420]:


for dataset in combined:
    dataset['IsAlone']=0
    dataset.loc[dataset['familysize']==1,'IsAlone']=1


# In[421]:


train[['IsAlone','Survived']].groupby('IsAlone',as_index=False).mean().sort_values(by='IsAlone')


# In[422]:


train=train.drop(['SibSp','Parch','familysize'],axis=1)


# In[423]:


test=test.drop(['SibSp','Parch','familysize'],axis=1)
combined=[train,test]


# In[424]:


freq=train.Embarked.dropna().mode()[0]
freq


# In[425]:


train['Embarked']=train['Embarked'].fillna(freq)


# In[426]:


train.info()


# In[429]:


train['Embarked']=train['Embarked'].map({'C':1,'S':2,'Q':3}).astype(int)
train.head()


# In[430]:


test.info()


# In[431]:


#there is one missing value in far in test set 
med=test['Fare'].median()
test['Fare']=test['Fare'].fillna(med)


# In[436]:


train['farerange']=pd.qcut(train['Fare'],4)
train[['farerange','Survived']].groupby('farerange',as_index=False).mean().sort_values(by='farerange')


# In[437]:


combined=[train,test] 
for dataset in combined:
    dataset.loc[dataset['Fare']<=7.91,'Fare']=1
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare']=2
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare']=3
    dataset.loc[(dataset['Fare']>31)&(dataset['Fare']<=512.329),'Fare']=4
    dataset['Fare']=dataset['Fare'].astype(int)


# In[444]:


train=train.drop('farerange',axis=1)


# In[441]:


test['Embarked']=test['Embarked'].map({'C':1,'S':2,'Q':3}).astype(int)
test.head()


# ## build our model

# In[448]:


X_train=train.drop('Survived',axis=1)
y_train=train['Survived']
X_train.shape,y_train.shape


# ## using logistic regression 

# In[ ]:


X_test=test
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred=log.predict(X_test)


# In[459]:


y_pred


# In[ ]:




