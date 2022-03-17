import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_linear.csv')

print(data.info())

X = data.values[:,0:1]
y = data.values[:,1:2]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Evaluate model
y_hat = linear_reg.predict(X_test)

print("Average loss:", np.sqrt(np.mean((y_hat-y_test)**2)))

# Draw
x0 = np.array([[30], [100]])
y0 = linear_reg.predict(x0)
plt.plot(x0, y0, 'r', linewidth=2)
plt.plot(X_train, y_train, 'b.')
plt.plot(X_test, y_test, 'y.')
plt.legend(['Linear line', 'Train dataset', 'Test dataset'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()