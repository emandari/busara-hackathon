import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# train.info()
# test.info()

train_columns = train.columns
# print(train_columns)

train = train.loc[:, train.isnull().mean() < 0.8]
# print([train.isnull().mean() < 0.8])
train = train.dropna(axis=1)
train_columns = train.columns
depressed = train['depressed']
surveyid = train['surveyid']

train = train.drop(columns=['surveyid', 'survey_date', 'depressed'])
train_columns = train.columns
# test['age'] = test[test['age'] == '.d']['age'].mean()
# print(test[test['age'] == '.d']['age'])

# Remove the '.d'
test['age'] = test['age'].replace('.d', test[test['age'] != '.d']['age'].median())
test['age'] = test['age'].fillna(test[test['age'] != '.d']['age'].median())
# test[test['age'] == '.d']['age'] = test[test['age'] != '.d']['age'].median()
# print(test[test['age'] == '.d']['age'].values)
# print(test[test['age'] != '.d']['age'].median())
# print("Mean: ", test['age'].describe())

test = test[train_columns]
X = train.values
y = depressed.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logisticRegressionModel = LogisticRegression()
logisticRegressionModel.fit(X_train, y_train)

# print(logisticRegressionModel.score(X_train, y_train))
# print(logisticRegressionModel.score(X_test, y_test))
# print(accuracy_score(y_test, logisticRegressionModel.predict(X_test)))

X_submit = test.values
print(logisticRegressionModel.predict(X_submit))
# print(test.isnull().mean())
