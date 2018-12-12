# Basic
import numpy as np
import pandas as pd

# ML and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("data/train.csv")
test = to_submit = pd.read_csv("data/test.csv")

# Data preprocessing
train = train.dropna(axis=1)
depressed = train['depressed']

train = train.drop(columns=['surveyid', 'survey_date', 'depressed'])

# Columns that will be used with test/submission set
train_columns = train.columns

# Assign values to X and y variables
X = train.values
y = depressed.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
logisticRegressionModel = LogisticRegression()
logisticRegressionModel.fit(X_train, y_train)

# print(logisticRegressionModel.score(X_train, y_train))
# print(logisticRegressionModel.score(X_test, y_test))
# print(accuracy_score(y_train, logisticRegressionModel.predict(X_train)))
# print(accuracy_score(y_test, logisticRegressionModel.predict(X_test)))

# Remove the '.d'
to_submit['age'] = to_submit['age'].replace('.d', to_submit[to_submit['age'] != '.d']['age'].median())

# Replace NaN values with median
to_submit['age'] = to_submit['age'].fillna(to_submit[to_submit['age'] != '.d']['age'].median())

to_submit = to_submit[train_columns]

X_submit = to_submit.values
y_submit = logisticRegressionModel.predict(X_submit)
test['depressed'] = pd.DataFrame(y_submit)
# test[['surveyid','depressed']].to_csv('submission.csv', index=False)
