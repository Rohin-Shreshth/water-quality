'''
-preprocess data
-train-test split our data
-build our model
-test our model
'''

import pandas as pd
from sklearn import preprocessing

# Read data
df = pd.read_csv('./datasets/water_potability.csv')
# print('data')

# Preprocess the data
df.dropna(inplace=True)

# Normalize data (convert all numbers to [0,1] range)
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

print(df)


