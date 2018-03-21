#A closer examination of the Ghouls, Goblins, and Ghosts... Boo! data from Kaggle. 

#Exploratory Data Analysis
##To begin we import all of the appropriate packages to be used throughout the course of the data analysis

import pandas as pd
import numpy as np
import seaborn as sns

##Import the dataset as a 'train' variable and run a quick overview
train = pd.read_csv("train.csv")
train.head()

##Begin exploratory data analysis to see trends in data
####A quick look determines that there are 1 identifier column, 6 distinct variables and 371 observations
####Variables include 'bone_length', 'rotting_flesh', 'hair_length', 'has_soul','color' and'type'
train.shape
train.columns.values


###Describe determines that there are four continuous variables, a factor variable and the classifier
###A closer examination shows that there is fairly even distribution among monster types
train.describe()
train['color'].value_counts()
####white    137
####clear    120
####green     42
####black     41
####blue      19
####blood     12

train['type'].value_counts()
####Ghoul     129
####Goblin    125
####Ghost     117

##A quick examination to determine whether the continuous variables will need to be transformed
###Variables appear to be normally distributed
sns.distplot(train['bone_length']) #Figure 1
sns.distplot(train['rotting_flesh']) #Figure 2
sns.distplot(train['hair_length']) #Figure 3
sns.distplot(train['has_soul']) #Figure 4

##A quick examination to determine whether the continuous variables cluster by type or color
###Prep column names
continuous_columns = list(train.columns.values)
continuous_columns.remove('id')

###Pairplots reveal that there doesn't seem to be much value to color, but there is clustering for type
sns.pairplot(train[continuous_columns], hue = 'color') #Figure 5
sns.pairplot(train[continuous_columns], hue = 'type') #Figure 6

###Examining color to hue type reveals no real value in including the color variable
sns.countplot(y="color", hue="type", data=train) #Figure 7

#Begin model building