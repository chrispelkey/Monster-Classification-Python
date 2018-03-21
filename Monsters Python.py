#A closer examination of the Ghouls, Goblins, and Ghosts... Boo! data from Kaggle. 

#Exploratory Data Analysis
##To begin we import all of the appropriate packages to be used throughout the course of the data analysis

import pandas as pd
import numpy as np
import seaborn as sns

##Import the dataset as a 'train' variable and run a quick overview
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
test.head()

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
##Import more packages for data prep and model building
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

##Start with data preparation
y = train["type"]
indexes = test["id"]

###Remove uneccessary variables
train = train.drop(["type","color","id"],axis=1)
test = test.drop(["color","id"],axis=1)

###Split training data into training and test data sets with 75% split to training
xtrain, xtest, ytrain, ytest = train_test_split(train, y, test_size=0.3, random_state=1127)

##See how logistic regression holds
logistic_model_1 = LogisticRegression(penalty='l2', C=1000)
logistic_model_1.fit(xtrain,ytrain)
ypred= logistic_model_1.predict(xtest) 

classification_report(ypred,ytest)
####             precision    recall  f1-score   support
####
####      Ghost       0.85      0.78      0.81        36
####      Ghoul       0.89      0.65      0.75        48
####     Goblin       0.43      0.68      0.53        28
####
####avg / total       0.76      0.70      0.71       112
