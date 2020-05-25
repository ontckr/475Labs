import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


dataset_all = pd.read_csv("CE 475 Spring 2020 Project Data.csv")
dataset = dataset_all.drop(dataset_all.index[100:120])

X = dataset[["x1", "x2", "x3", "x4", "x5", "x6"]]
y = dataset["Y"]
names = ["x1", "x2", "x3", "x4", "x5", "x6"]

regression = LinearRegression()
rfe = RFE(regression, n_features_to_select=1)
rfe.fit(X, y)

print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
