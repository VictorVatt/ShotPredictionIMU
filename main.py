from mat4py import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = loadmat("data/new_data_all_7.mat")

data = np.array(data["all_data"], dtype=object)

N_sujet = 0
N_coup = 0


data = data[N_sujet][0][N_coup][0]




df = pd.DataFrame(data, columns=["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"])



print(df)
