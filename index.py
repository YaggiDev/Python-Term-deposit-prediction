from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from sklearn import metrics
from IPython.display import Image
from sklearn import tree
import pydotplus
import matplotlib as mpl
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.externals import joblib

# -*- coding: utf-8 -*-
# Dataset source: https://archive.ics.uci.edu/ml/datasets/bank+marketing

columns = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance', \
           'Housing', 'Loan', 'Contact', 'Day', 'Month', 'Duration', \
           'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Target']
data = pd.read_csv('bank-full.csv', sep=';', names=columns, header=1)
Y = pd.DataFrame()
X = pd.DataFrame()
Y['Target'] = data['Target']
print(Y)
X = data.drop(['Target'], 1)
print(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)
train = pd.concat([x_train, y_train], axis=1, sort=False)
test = pd.concat([x_test, y_test], axis=1, sort=False)
train_test_data = [train, test]
print(train.columns)

print(y_test)

print(train.dtypes)
print(train_test_data[0].shape)
print("Null values: ", train.isnull().sum(axis=0))
print(train_test_data[0].dtypes)
print(train_test_data[0]['Target'].value_counts())

# Answers mapping 0 - no, 1 - yes
answer_mapping = {"no": 0, "yes": 1}
for dataset in train_test_data:
    dataset['Target'] = dataset['Target'].map(answer_mapping)

# Mapping Marital
print(train_test_data[0]['Marital'].value_counts())
marital_mapping = {"single": 1, "married": 2, "divorced": 3, "uknown": 0}
for dataset in train_test_data:
    dataset['Marital'] = dataset['Marital'].map(marital_mapping)

# Mapping Education
print(train_test_data[0]['Education'].value_counts())
education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
for dataset in train_test_data:
    dataset['Education'] = dataset['Education'].map(education_mapping)

# Mapping Default
print(train_test_data[0]['Default'].value_counts())
default_mapping = {"no": 0, "yes": 1}
for dataset in train_test_data:
    dataset['Default'] = dataset['Default'].map(default_mapping)

# Balance
print(train_test_data[0]['Balance'].describe())
sns.boxplot(train_test_data[0]['Balance'])
plt.title('Balance boxplot')
plt.show()
print(train_test_data[0][(train_test_data[0].Balance > 80000)].Age.count())
print(train_test_data[0].shape)
train_test_data[0].drop(train_test_data[0][train_test_data[0]['Balance'] > 80000].index,
                                           inplace=True)
print(train_test_data[0].shape)



def bar_chart(feature):
    accepted = train_test_data[0][train_test_data[0]['Target'] == 1][feature].value_counts()
    declined = train_test_data[0][train_test_data[0]['Target'] == 0][feature].value_counts()
    df = pd.DataFrame([accepted, declined])
    df.index = ['Accepted', 'Declined']

    df.plot(kind='bar', stacked='True', figsize=(10, 10))
    plt.title(feature)
    plt.savefig('Diagrams/' + feature + "_Diagram.png")
    plt.show()


def data_split():
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_indexes = round(0.75 * len(train_test_data) - 1)
    all_indexes = range(len(train_test_data))
    train_ix = []
    sample = random.sample(all_indexes, train_indexes)
    print(len(sample))


print(train_test_data[0].Job.value_counts())
bar_chart('Loan')


def sns_plot(feature, max=0, min=0):
    facet = sns.FacetGrid(train_test_data[0], hue="Target", aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(0, train_test_data[0][feature].max()))
    facet.add_legend()
    if max == 0:
        max = train_test_data[0][feature].max()
    if min == 0:
        min = train_test_data[0][feature].min()
    plt.xlim(min, max)
    plt.savefig('Diagrams/' + feature + "_Diagram.png")
    plt.show()


# Job mapping
# job_mapping = ["unknown", "unemployed", "self-employed", "student", "blue-collar", "housemaid", "entrepreneur",
#                "management", "services", "technician", "admin", "retired"]

for i in range(2):
    train_test_data[i] = pd.concat([train_test_data[i], pd.get_dummies(train_test_data[i]['Job'])], axis=1)
print(train_test_data[0].columns)

print("Min age: ", train_test_data[0].Age.min(), "\nMax age: ", train_test_data[0].Age.max())

# Contact mapping
print(train_test_data[0].Contact.value_counts())
contact_mapping = {"unknown": 0, "cellular": 1, "telephone": 2}
for dataset in train_test_data:
    dataset['Contact'] = dataset['Contact'].map(contact_mapping)

# Loan mapping - has personal loan?
print(train_test_data[0].Loan.value_counts())
loan_mapping = {"no": 0, "yes": 1}
for dataset in train_test_data:
    dataset.Loan = dataset.Loan.map(loan_mapping)

# Housing mapping - has housing loan?
print(train_test_data[0].Housing.value_counts())
for dataset in train_test_data:
    dataset.Housing = dataset.Housing.map(answer_mapping)

# Poutcome mapping
print(train_test_data[0].Poutcome.value_counts())
poutcome_mapping = {"failure": 0, "nonexistent": 1, "success": 2, "unknown": 3, "other": 4}
for dataset in train_test_data:
    dataset.Poutcome = dataset.Poutcome.map(poutcome_mapping)
# Month mapping
# TODO change categorical to binary variables
print(train_test_data[0].Month.value_counts())
for i in range(2):
    train_test_data[i] = pd.concat([train_test_data[i], pd.get_dummies(train_test_data[i]['Month'])], axis=1)
month_mapping = {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9,
                 "nov": 10, "dec": 11}
for dataset in train_test_data:
    dataset.Month = dataset.Month.map(month_mapping)

fig, ax = plt.subplots()
ind = np.arange(12)
width = 0.35
ax.bar(ind, train_test_data[0].loc[train_test_data[0]['Target'] == 1, 'Month'].value_counts(), width,
       label='Target = 1')
ax.bar(ind + width, train_test_data[0].loc[train_test_data[0]['Target'] == 0, 'Month'].value_counts(), width,
       label='Target = 0')
ax.set_title('Grouped value counts in months')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(month_mapping)
ax.legend()
ax.autoscale_view()
plt.savefig('Diagrams/MonthsCounts.png')
plt.show()

print(train_test_data[0].head(10))
print(train_test_data[0].dtypes)
print(train_test_data[0].shape)
print(train_test_data[0].isnull().sum(axis=0))

# Groupin Age
for dataset in train_test_data:
    dataset.loc[dataset['Age'] < 26, 'Age'] = 0,
    dataset.loc[(dataset['Age'] >= 26) & (dataset['Age'] < 40), 'Age'] = 1,
    dataset.loc[(dataset['Age'] >= 40) & (dataset['Age'] < 60), 'Age'] = 2,
    dataset.loc[(dataset['Age'] >= 60), 'Age'] = 3

print(train_test_data[0].iloc[5:])
sns_plot('Marital')
print(train_test_data[0].Age.value_counts())

group_one = pd.DataFrame()
group_one = train_test_data[0].loc[train_test_data[0]['Target'] == 1, 'Age'].value_counts().rename_axis('Age '
                                                                                                        'group').reset_index(
    name='Counts')
group_one['Target'] = 1
group_one['Counts'] = group_one['Counts'] / group_one['Counts'].sum()
group_zero = pd.DataFrame()
group_zero = train_test_data[0].loc[train_test_data[0]['Target'] == 0, 'Age'].value_counts().rename_axis('Age '
                                                                                                         'group').reset_index(
    name='Counts')
group_zero['Target'] = 0
group_zero['Counts'] = group_zero['Counts'] / group_zero['Counts'].sum()

group = pd.DataFrame()
group = pd.concat([group_one, group_zero])
print(group)

g = sns.catplot(x='Age group', y='Counts', col='Target', palette="ch:.25", kind='bar', data=group)
g.set_ylabels('Counts percentage [%]', fontsize=15)
g.set_xlabels('Age groups', fontsize=15)
g.set_yticklabels(fontsize=15)
g.set_xticklabels(fontsize=15)
# g.set_titles("Target = {col_name}", fontsize = 20)
plt.savefig('Diagrams/AgeByGroup_Diagram.png')
plt.show()
bar_chart('Age')


# seasoning
# sns_plot('Month')

# scatter_matrix(train.iloc[:,0:5])
# scatter_matrix(train[train.columns[0:5]])
# plt.show()
# for column in train:
#     print(train[column].describe())


def Normalize(feature):
    for dataset in train_test_data:
        col = dataset[[feature]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        col_scaled = min_max_scaler.fit_transform(col)
        _feature = feature + "_norm"
        dataset[_feature] = pd.DataFrame(col_scaled)
    print("Normalized {}: ".format(feature), X[_feature].tail(10))


temp = pd.DataFrame()
temp = train_test_data[0].corr(method='pearson')
# temp.to_csv('corr.csv')

# Pearson Correlation
figure = plt.figure(figsize=(25, 17))
cor = train_test_data[0].corr()
g = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2g', linewidths=1, cbar_kws={}, square=True)
g.set_title('Correlation diagram', fontsize=30)
g.set_ylim(28, 0)
plt.tight_layout()
# g.figure.axes[-1].yaxis.label.set_size(20)
# plt.savefig('Diagrams/CorrelationHeatMap_Diagram.png')
plt.show()

# building target array
train_test_data[0] = train_test_data[0].drop('Duration', axis=1)
train_test_data[1] = train_test_data[1].drop('Duration', axis=1)

target_train = train_test_data[0]['Target']
target_test = train_test_data[1]['Target']
train_test_data[0] = train_test_data[0].drop('Target', axis=1)
train_test_data[1] = train_test_data[1].drop('Target', axis=1)

train_test_data[0] = train_test_data[0].drop('Job', axis=1)
train_test_data[1] = train_test_data[1].drop('Job', axis=1)
train_test_data[0] = train_test_data[0].drop('Month', axis=1)
train_test_data[1] = train_test_data[1].drop('Month', axis=1)

print(train_test_data[0].columns)
print(train_test_data[0].head(5))


# Save the model
def save_model(model, filename, path="Models/"):
    filename = path + filename + ".joblib"
    joblib.dump(model,filename)


# Load the model
def load_model(filename, path="Models/"):
    filename = path + filename + ".joblib"
    return joblib.load(filename)


# Check pickle if better
def check_model(model, filename, x_test, y_test, path="Models/"):
    new_model_score = model.score(x_test, y_test)
    print(f"Method: {filename}\n")
    print("New model\nScore: ",new_model_score,"\n__")
    saved_model = load_model(filename, path)
    saved_model_score = saved_model.score(x_test, y_test)
    print("Saved model\nScore: ", saved_model_score, "\n__")
    if new_model_score > saved_model_score:
        save_model(model, filename)
        return model
    else:
        return saved_model


cov = train_test_data[0].cov()

knn = KNeighborsClassifier(algorithm='auto', metric='mahalanobis', metric_params={"V": cov},
                           n_neighbors=3,
                           weights='distance')
# knn = KNeighborsClassifier(algorithm='auto', metric='euclidean',
#                            n_neighbors=3,
#                            weights='distance')
knn.fit(train_test_data[0], target_train)
# pickle_in(knn,"KNNModel")
# joblib.dump(knn, 'Models/KNNModel.joblib')
knn = check_model(knn, "KNNModel", train_test_data[1], target_test)
# save model

# load model
# knn = joblib.load('file')
pred_knn = knn.predict(train_test_data[1])
print(pred_knn)  # Classifing
pred_prob_knn = knn.predict_proba(train_test_data[1])
print(pred_prob_knn)  # probability

print("KNN score: ", knn.score(train_test_data[1], target_test))
# Importing for SAS
train_test_data[0].to_csv('train_test_data.csv')
train_test_data[1].to_csv('test.csv')

# # KNN plotting
# X = train_test_data[0].to_numpy()
# Y = target_train.to_numpy()
# print(X)
# plot_decision_regions(X, Y, clf = knn, legend = 2)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('KNN with K = 3')
# plt.show()

clf = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=0.01)
clf.fit(train_test_data[0], target_train)
# joblib.dump(clf, 'Models/DecisionTree.joblib')
clf = check_model(clf, "DecisionTree", train_test_data[1], target_test)
print(clf.predict_proba(train_test_data[1]))
print("Decision tree score: ", clf.score(train_test_data[1], target_test))
tree.export_graphviz(clf,out_file='Diagrams/DecisionTree.dot',class_names=True, label='all',
                     rounded=True, filled = True)

print("Train shape: ", train_test_data[0].shape)
print("Test shape: ", train_test_data[1].shape)

reg = LogisticRegression( solver = 'lbfgs', max_iter= 10000)
reg.fit(train_test_data[0], target_train)
# joblib.dump(reg, 'Models/LogisticRegression.joblib')
reg = check_model(reg, "LogisticRegression", train_test_data[1], target_test)
print(reg.predict(train_test_data[1]))
print("Logistic regression score: ", reg.score(train_test_data[1], target_test))
print("Logistic regression coefficients: ", reg.coef_)

# fpr, tpr, threshold = metrics.roc_auc_score(target_test,pred_prob_knn[1])
# print(metrics.auc(fpr,tpr))

# dot_data = StringIO()
# export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters= True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# MLP Classifier
mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(13, 2))
mlp.fit(train_test_data[0], target_train)
# joblib.dump(mlp, 'Models/MLPClassifier.joblib')
mlp = check_model(mlp, "MLPClassifier", train_test_data[1], target_test)
print("Neural network MLP: ", mlp.score(train_test_data[1], target_test))
