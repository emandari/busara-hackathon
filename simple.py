import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")

# Drop all columns with NaN
train = train.dropna(axis=1)

# # Drop NaN Values with more than 50%
# train = train.loc[:, train.isnull().mean() < 0.5]

# Impute NaN values with most frequent
# imputer = SimpleImputer(strategy="most_frequent")
# train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)

X = train.drop(columns=['surveyid', 'survey_date', 'depressed', 'wage_expenditures', 'ent_nonag_revenue'])
y = train.depressed

# print(X.shape, y.shape)
# print(y.value_counts())

# Machine Learning
X = X.values
y = y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
model = LogisticRegression()
model.fit(X, y)

# Test Accuracy
print(accuracy_score(y_test, model.predict(X_test)))
