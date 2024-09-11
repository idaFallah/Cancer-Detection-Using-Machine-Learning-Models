import numpy as np #arrays
import pandas as pd #data science and data frames?
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization

dataset = pd.read_csv('/content/data.csv')

dataset.head()

dataset.shape

dataset.info()

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

dataset.select_dtypes(include=['float64', 'int64']).columns

len(dataset.select_dtypes(include=['float64', 'int64']).columns)

#statistical summary
dataset.describe()

dataset.columns

dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[dataset.isnull().any()]

len(dataset.columns[dataset.isnull().any()])

dataset['Unnamed: 32'].count()

dataset = dataset.drop(columns='Unnamed: 32')

dataset.isnull().values.any()

dataset.select_dtypes(include='object').columns

dataset['diagnosis'].unique()

dataset['diagnosis'].nunique()

#one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

dataset.head()

sns.countplot(dataset['diagnosis_M'], label='Count')
plt.show()

#count of B(0) values
(dataset.diagnosis_M == 0).sum()

#count of M(1) values
(dataset.diagnosis_M == 1).sum()

dataset_2 = dataset.drop(columns= 'diagnosis_M')

dataset_2.head()

dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize = (20, 10), title = 'Correlated with diagnosis_M', rot=45, grid=True
)

#correlatin matrix
corr = dataset.corr()

corr

#heatmap to analyze the correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)

#split the dataset into train/test set

dataset.head()

#matrix of features/ Independent variables
x = dataset.iloc[:, 1:-1].values

x.shape

#target variable/ dependant features
y = dataset.iloc[:, -1].values

y.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

#feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

#building the model

#logistic regression
from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state=0) #built an instance of LR class

classifier_lr.fit(x_train, y_train) #tarined the model

y_pred = classifier_lr.predict(x_test) #predicted the vlaues

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

#defining variables to analyze model's performance
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
                       columns=['Model', 'Accuracy', 'F1 score', 'Precision', 'Recall'])

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score #cross validation

accuracies = cross_val_score(estimator=classifier_lr, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#random forest
from sklearn.ensemble import RandomForestClassifier

classifier_rm = RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train, y_train)

y_pred = classifier_rm.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Random forest', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = results._append(model_results, ignore_index=True)

results

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score #cross validation

accuracies = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#randomized search to find the best params for LR
from sklearn.model_selection import RandomizedSearchCV

parameters = {'penalty':['l1', 'l2', 'elasticnet', 'None'],
              'C':[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] #????in adada az koja mian?
              ,
              'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']




}

parameters

random_search = RandomizedSearchCV(estimator=classifier_lr, param_distributions=parameters, n_iter=10, scoring='roc_auc', n_jobs=-1, cv=10, verbose=3)

random_search.fit(x_train, y_train)

random_search.best_estimator_

random_search.best_score_

random_search.best_params_

#finalize model as LR

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.75, random_state=0, solver='sag') #built an instance of LR class
classifier.fit(x_train, y_train) #tarined the model

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Final Logistic Regression', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = results._append(model_results, ignore_index=True)

results

#cross validation for final model
from sklearn.model_selection import cross_val_score #cross validation

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10) #to increase accuracy

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

#predicting a single observation

dataset.head()

single_obs = [[17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760,	0.3001,	0.14710, 0.2419, 0.07871, 1.0950, 0.9053, 8.589, 153.40, 0.006399, 0.04904,	0.05373, 0.01587, 0.03003, 0.006193, 25.38,
17.33, 184.60, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890]]

single_obs

classifier.predict(sc.transform(single_obs))



