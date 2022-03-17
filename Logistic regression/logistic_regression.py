import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv('data_Logistic.csv')

print(data.info())

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

# Data preprocessing
labelEncoder_X = LabelEncoder()
X.iloc[:,0] = labelEncoder_X.fit_transform(X.iloc[:,0])
X = StandardScaler().fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, data['Purchased'], test_size=0.2, random_state=0)

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)

# Evaluate model
y_hat = logistic_reg.predict(X_test)
print('\nThe mean accuracy: ', logistic_reg.score(X_test,y_test))
print('Classification Report:\n', classification_report(y_test, y_hat))