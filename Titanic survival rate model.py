#!/usr/bin/env python
# coding: utf-8

# Titanic survival rate using Logistic regression

# In[77]:


#Lets first import the required libraries
import pandas as pd
from sklearn import preprocessing


# Load data from CSV file

# In[78]:


df = pd.read_csv(r"C:\Users\Anne Marie\Documents\train.csv")
test =  pd.read_csv(r"C:\Users\Anne Marie\Documents\test.csv")


# In[79]:


print(df.shape)
print(test.shape)


# In[80]:


df.columns


# In[81]:


test.columns


# Lets see if there are any null values in our dataset

# In[82]:


df.isna().sum()


# In[83]:


test.isna().sum()


# Data pre-processing and selection
# 
# Lets select some features for modeling, and exclude any rows with null values Lets also change some datatypes. 

# In[121]:


df = df.loc[df.Embarked.notna(),['Survived', 'Pclass', 'Sex', 'Embarked' ]]
test = test.loc[:,['Pclass', 'Sex', 'Embarked']]


# In[85]:


print(df.shape)
print(test.shape)


# In[86]:


df.head()


# In[87]:


test.head()


# In[88]:


x = df.loc[:, ['Pclass']]
y = df.Survived


# In[89]:


x.shape


# Let us now convert the sex and embarked feature to numerical form to make it easy for our algorithm to use

# In[90]:


y.shape


# Lets use Logistic regression because it is good for binary classification

# In[91]:


from sklearn.linear_model import LogisticRegression


# In[92]:


logreg =  LogisticRegression(solver ='lbfgs')


# Lets evaluate our model. Here we are cross validating our logreg model using one feature which is the Pclass. We will use 5-fold cross validation. Our output is the mean accuracy of the 5-fold cross validations.

# In[93]:


from sklearn.model_selection import cross_val_score


# In[99]:


cross_val_score(logreg, x, y, cv=5, scoring='accuracy').mean()


# Lets check how this compares to the null accuracy- The accuracy we'll get by predicting the most frequent class. This is just an optional step

# In[100]:


y.value_counts(normalize=True)


# Lets convert our categorical features into numerical values

# In[101]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)


# In[102]:


x= df.drop('Survived', axis='columns')


# In[103]:


x.head()


# Lets specify which columns to encode using column transformer. In this case, we transform the sex and embarked columns, while passing through the Pclass column

# In[104]:


from sklearn.compose import make_column_transformer


# In[105]:


column_trans = make_column_transformer(
    (OneHotEncoder(), ['Sex', 'Embarked']),
    remainder='passthrough')


# In[106]:


column_trans.fit_transform(x)


# In[107]:


test.head()


# In[108]:


x_test= test
column_trans.fit_transform(x_test)


# In[109]:


from sklearn.pipeline import make_pipeline


# In[110]:


#pipeline is for chaining steps together. So in this case, our pipeline transforms 
#our specified columns, and then in builds our logreg model.
pipe = make_pipeline(column_trans,logreg)


# In[111]:


cross_val_score(pipe, x, y, cv=4, scoring= 'accuracy').mean()


# In[112]:


#Our accuracy has improved to 0.77


# Lets use our test data set to pass the model

# In[113]:


x_test = x_test.sample(5, random_state=99)
x_test


# In[114]:


pipe.fit(x, y)


# In[115]:


pipe.predict(x_test)


# In[ ]:




