import numpy as np
import csv
import pandas as pd
import matplotlib as plt
import sys
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)


dataset_all = pd.read_csv("CE 475 Spring 2020 Project Data.csv")
dataset = dataset_all.drop(dataset_all.index[100:120])
dataset_predict = dataset_all.drop(dataset_all.index[0:100])

X = dataset[["x1", "x3"]]
y = dataset["Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
dfPred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(dfPred)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("---Predict Unlabeled Data---")
final_pred = regressor.predict(dataset_predict[["x2", "x3"]])
dfPred = pd.DataFrame({'x2': dataset_predict["x2"], 'x3': dataset_predict["x3"], 'Predicted': final_pred})
print(dfPred)

