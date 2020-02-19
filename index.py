from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix

# -*- coding: utf-8 -*-
# Dataset source: https://archive.ics.uci.edu/ml/datasets/bank+marketing

columns = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance',\
           'Housing', 'Loan', 'Contact', 'Day', 'Month', 'Duration',\
           'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Target']
train = pd.read_csv('bank-full.csv',sep=';',names=columns,header=1)
test = pd.read_csv('bank.csv', sep=';', names=columns,header=1)
train_test_data = [train,test]
print(train.columns)

print(train.dtypes)

print(train.isnull().sum(axis = 0))
print(train.dtypes)
print(train['Target'].value_counts())

# Answers mapping 0 - no, 1 - yes
answer_mapping = {"no": 0, "yes" : 1}
for dataset in train_test_data:
    dataset['Target'] = dataset['Target'].map(answer_mapping)

# Mapping Marital
print(train['Marital'].value_counts())
marital_mapping = {"single":0, "married":1, "divorced":2}
for dataset in train_test_data:
    dataset['Marital'] = dataset['Marital'].map(marital_mapping)

# Mapping Education
print(train['Education'].value_counts())
education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
for dataset in train_test_data:
    dataset['Education'] = dataset['Education'].map(education_mapping)

# Mapping Default
print(train['Default'].value_counts())
default_mapping = {"no": 0, "yes": 1}
for dataset in train_test_data:
    dataset['Default'] = dataset['Default'].map(default_mapping)

def bar_chart(feature):
    accepted = train[train['Target']==1][feature].value_counts()
    declined = train[train['Target']==0][feature].value_counts()
    df = pd.DataFrame([accepted,declined])
    df.index = ['Accepted','Declined']
    df.plot(kind='bar',stacked='True', figsize = (10,10))
    plt.title(feature)
    plt.show()

def data_split():
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_indexes = round(0.75*len(train_test_data)-1)
    all_indexes = range(len(train_test_data))
    train_ix = []
    sample = random.sample(all_indexes,train_indexes)
    print(len(sample))

print(train.Job.value_counts())
bar_chart('Loan')
def sns_plot(feature, max = 0, min = 0):
    facet = sns.FacetGrid(train, hue="Target", aspect = 4)
    facet.map(sns.kdeplot, feature, shade = True)
    facet.set(xlim=(0,train[feature].max()))
    facet.add_legend()
    if max == 0:
        max = train[feature].max()
    if min == 0:
        min = train[feature].min()
    plt.xlim(min,max)
    plt.show()

# Job mapping
job_mapping = {"unknown": 0, "unemployed": 1, "self-employed": 2, "student": 3, "blue-collar": 4, "housemaid": 5, "entrepreneur": 6, "management": 7, "services": 8, "technician": 9, "admin.": 10, "retired": 11}
for dataset in train_test_data:
    dataset['Job'] = dataset['Job'].map(job_mapping)
print(train.head(20))


print("Min age: ",train.Age.min(),"\nMax age: ",train.Age.max())
sns_plot('Age')

# Contact mapping
print(train.Contact.value_counts())
contact_mapping = {"unknown": 0, "cellular": 1,"telephone": 2}
for dataset in train_test_data:
    dataset['Contact'] = dataset['Contact'].map(contact_mapping)

# Loan mapping - has personal loan?
print(train.Loan.value_counts())
loan_mapping = {"no": 0,"yes": 1}
for dataset in train_test_data:
    dataset.Loan = dataset.Loan.map(loan_mapping)

# Housing mapping - has housing loan?
print(train.Housing.value_counts())
for dataset in train_test_data:
    dataset.Housing = dataset.Housing.map(answer_mapping)

# Poutcome mapping
print(train.Poutcome.value_counts())
poutcome_mapping = {"failure": 0,"nonexistent": 1,"success": 2, "unknown": 3, "other": 4}
for dataset in train_test_data:
    dataset.Poutcome = dataset.Poutcome.map(poutcome_mapping)

# Month mapping
print(train.Month.value_counts())
month_mapping = {"jan": 0, "feb": 1,"mar": 2,"apr": 3,"may": 4,"jun": 5,"jul": 6,"aug": 7,"sep": 8,"oct": 9,"nov": 10,"dec": 11}
for dataset in train_test_data:
    dataset.Month = dataset.Month.map(month_mapping)

print(train.head(10))
print(train.dtypes)
print(train.shape)
print(train.isnull().sum(axis = 0))

#Groupin Age
for dataset in train_test_data:
     dataset.loc[ dataset['Age']<28, 'Age'] = 0,
     dataset.loc[ (dataset['Age']>=28) & (dataset['Age']<40), 'Age'] = 1,
     dataset.loc[ (dataset['Age']>=40) & (dataset['Age']<58), 'Age']= 2,
     dataset.loc[ (dataset['Age']>=58),'Age'] = 3
bar_chart('Age')

sns_plot('Marital')
sns_plot('Age',4,-1)

#seasoning
sns_plot('Month')

#scatter_matrix(train.iloc[:,0:5])
# scatter_matrix(train[train.columns[0:5]])
# plt.show()
for column in train:
    print(train[column].describe())


def Normalize(feature):
    for dataset in train_test_data:
        col = dataset[[feature]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        col_scaled = min_max_scaler.fit_transform(col)
        _feature = feature +"_norm"
        dataset[_feature] = pd.DataFrame(col_scaled)

    print("Normalized {}: ".format(feature), train[_feature].tail(10))

# Campaign normalization
Normalize('Campaign')
# Pdays normalization
Normalize('Pdays')


print(train.shape)
#building target array
train = train.drop('Duration',axis=1)
test = test.drop('Duration',axis=1)
target_train = train['Target']
target_test = test['Target']
train = train.drop('Target',axis=1)
test = test.drop('Target',axis=1)

print(train.shape)

knn = KNeighborsClassifier(algorithm='auto',metric='minkowski',n_neighbors=3,weights='uniform')
knn.fit(train,target_train)
print(knn.predict(test)) # Classifing
print(knn.predict_proba(test)) # probability

print("KNN score: ",knn.score(test,target_test))
train.to_csv('train_test_data.csv')
test.to_csv('test.csv')
