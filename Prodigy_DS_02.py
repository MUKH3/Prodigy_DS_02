#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


# Load the dataset
titanic_df = pd.read_csv("C:/Users/Trishit/Documents/Prodigy Infotech/Data Science/Task 2/titanic.csv")


# DATASET DESCRIPTION:
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[11]:


# Display the dataset
titanic_df


# In[12]:


# Structure of the dataset
titanic_df.shape


# In[13]:


# Check for missing values
print(titanic_df.isnull().sum())


# Now we have to handle the null values.

# In[14]:


# Fill missing values in 'Age' column with median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' column with the mode
mode_embarked = titanic_df['Embarked'].mode()[0]
titanic_df['Embarked'].fillna(mode_embarked, inplace=True)

# Drop the 'Cabin' column due to a large number of missing values
titanic_df.drop(columns=['Cabin'], inplace=True)


# In[15]:


# Check if missing values have been handled
print(titanic_df.isnull().sum())


# Now that missing values have been handled, we can proceed with exploratory data analysis (EDA) to gain insights into the dataset. Let's start by visualizing the distribution of some key variables and exploring relationships between variables.

# In[23]:


titanic_df


# In[16]:


# Visualize the distribution of survival
sns.countplot(x='Survived', data=titanic_df)
plt.title('Survival Distribution')
plt.show()


# In[ ]:





# In[17]:


# Visualize the survival distribution by sex
sns.countplot(x='Survived', hue='Sex', data=titanic_df)
plt.title('Survival Distribution by Sex')
plt.show()


# In[ ]:





# In[18]:


# Visualize the survival distribution by passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
plt.title('Survival Distribution by Passenger Class')
plt.show()


# In[ ]:





# In[28]:


# Visualize the age distribution using a histogram
plt.figure(figsize=(10, 6))
plt.hist(titanic_df['Age'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()


# In[ ]:





# In[20]:


# Explore the relationship between age and survival
sns.boxplot(x='Survived', y='Age', data=titanic_df)
plt.title('Age vs Survival')
plt.show()


# In[ ]:





# In[29]:


# Age distribution by passenger class and sex
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', hue='Sex', data=titanic_df)
plt.title('Age Distribution by Passenger Class and Sex')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()


# In[ ]:





# In[30]:


# Fare distribution by passenger class and embarked port
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', hue='Embarked', data=titanic_df)
plt.title('Fare Distribution by Passenger Class and Embarked Port')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# In[ ]:





# In[31]:


# Survival rate by age and passenger class
plt.figure(figsize=(10, 6))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=titanic_df, split=True)
plt.title('Survival Rate by Age and Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()


# In[ ]:





# In[35]:


# Exclude non-numeric columns
numeric_columns = titanic_df.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr = numeric_columns.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, cbar=False)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:





# In[33]:


# Create a new feature representing family size
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# Survival rate by family size
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=titanic_df)
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.show()


# In[ ]:





# In[ ]:




