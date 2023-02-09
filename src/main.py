from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

cols_labels = ["AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax", "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax", "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
               "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax", "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax", "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"]

frequence = 60
new_shot_data = pd.read_csv("../data/test_shots.csv", skiprows=range(0,11), usecols=range(2,8))
new_shot_data.columns = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]

AccX_peaks = find_peaks(new_shot_data["AccX"], height=1, threshold=1, distance=60)
peaks = AccX_peaks[0]

NB_shots = len(AccX_peaks[0])

to_predict_shot = pd.DataFrame(columns=cols_labels)

for i in peaks:
    data_new_shot = new_shot_data[i-frequence:i+frequence]
    row = list()
    for j in range(6):
        mean = np.mean(data_new_shot.iloc[:, j])
        sd = np.std(data_new_shot.iloc[:, j])
        skewness = skew(data_new_shot.iloc[:, j])
        kurtosisness = kurtosis(data_new_shot.iloc[:, j])
        minimum = np.min(data_new_shot.iloc[:, j])
        maximum = np.max(data_new_shot.iloc[:, j])
        row.append(mean)
        row.append(sd)
        row.append(skewness)
        row.append(kurtosisness)
        row.append(minimum)
        row.append(maximum)
    to_predict_shot.loc[len(to_predict_shot)] = row

print(to_predict_shot)

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

typeofshot = clf.predict(to_predict_shot)
print(typeofshot)

