import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Network_Ads.csv')
print(data.info())

data.iloc[:,[2]] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data.iloc[:, [2]])

X = data.iloc[:,:3].values
X[:,0] = LabelEncoder().fit_transform(X[:,0])
y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def weight(distances):
    return np.exp(-distances**2/.5)

clf = KNeighborsClassifier(n_neighbors=3, weights=weight)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Precision:', accuracy_score(y_test, y_pred))