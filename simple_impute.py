import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from xgboost import XGBClassifier

train = pd.read_csv("data/train.csv")

# Drop NaN Values with more than 50%
# XGboost becomes for accurate with when value is 0.8 but overfits
train = train.loc[:, train.isnull().mean() < 0.5]


X = train.drop(columns=['surveyid', 'survey_date', 'depressed'])
y = train.depressed

# Impute NaN values with most frequent
imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(X)

# Machine Learning
# X = X.tolist()
y = y.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# model = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
model = LogisticRegression()
model.fit(X_train, y_train)

classifier = XGBClassifier(learning_rate=0.1, max_depth=4, min_samples_split=100)
classifier.fit(X_train, y_train)

classifier1 = XGBClassifier(learning_rate=0.1, max_depth=4, min_samples_split=100)
classifier1.fit(X, y)

m = RFECV(RandomForestClassifier(), scoring='accuracy')
m.fit(X_train, y_train)

# Test Accuracy
print("#######")
print(accuracy_score(y_test, model.predict(X_test)))
print(accuracy_score(y_train, model.predict(X_train)))

print("#######")
print(accuracy_score(y_test, classifier.predict(X_test)))
print(accuracy_score(y_train, classifier.predict(X_train)))


print("#######")
print(accuracy_score(y, classifier.predict(X)))
print(accuracy_score(y, model.predict(X)))


print("####### Entire Set")
print(accuracy_score(y, classifier1.predict(X)))
# print(accuracy_score(y, model.predict(X)))


print("#######")
print(accuracy_score(y, classifier.predict(X)))
print(accuracy_score(y, model.predict(X)))

print("#### RFECV ###", accuracy_score(y_test, m.predict(X_test)))
print("#### XGBoost ###", accuracy_score(y_test, classifier.predict(X_test)))
print("#### LogisticRegression ###", accuracy_score(y_test, model.predict(X_test)))

