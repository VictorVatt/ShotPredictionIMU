import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

data = pd.read_csv("../data/training_dataset")
data.drop(457, axis=0, inplace=True)
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))


new_shot = [25.868747, 28.013614, 1.536513, 1.847528, 0.559863, 121.203224, 12.073428, 16.164174, -0.084047, -0.352892, -29.179884, 53.120020, -8.984323, 24.587130, -1.123868, 15.838077, -156.800000, 109.393456, -53.493653, 174.463880, -1.211994, 1.293752, -599.426300, 256.896970, 203.870238, 393.012398, 0.659535, -0.413649, -291.259770, 1225.891100, -70.000000, 176.142779, -0.551873, -1.159833, -420.166020, 129.943850]


new_stoke = pd.DataFrame(columns=cols_labels)
new_stoke.loc[0] = new_shot

typeofshot = clf.predict(new_stoke)
print(typeofshot)